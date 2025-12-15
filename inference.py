import argparse
import glob
import numpy as np
import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from huggingface_hub import login
from omegaconf import OmegaConf
from os.path import basename, splitext, join
from PIL import Image
from torch.amp import autocast
from typing import Optional, List

from src import StereoSpace

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

if "HF_TOKEN_LOGIN" in os.environ:
    login(token=os.environ["HF_TOKEN_LOGIN"])


def process_image(
    src_path: str,
    src_intrinsics: Optional[torch.Tensor] = None,
    tgt_intrinsics: Optional[torch.Tensor] = None,
    crop_size: int = 768,
) -> tuple[torch.Tensor, torch.Tensor]:
    src_img = Image.open(src_path).convert("RGB")
    w, h = src_img.size

    scale = crop_size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    src_resized = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    top = (new_h - crop_size) // 2
    left = (new_w - crop_size) // 2

    if src_intrinsics is not None:
        src_K = src_intrinsics.clone().float()
        src_K[0, 0] *= scale  # fx
        src_K[1, 1] *= scale  # fy
        src_K[0, 2] *= scale  # cx
        src_K[1, 2] *= scale  # cy

        src_K[0, 2] -= left
        src_K[1, 2] -= top
        src_K = src_K.unsqueeze(0)
    else:
        src_K = None

    if tgt_intrinsics is not None:
        tgt_K = tgt_intrinsics.clone().float()
        tgt_K[0, 0] *= scale  # fx
        tgt_K[1, 1] *= scale  # fy
        tgt_K[0, 2] *= scale  # cx
        tgt_K[1, 2] *= scale  # cy

        tgt_K[0, 2] -= left
        tgt_K[1, 2] -= top
        tgt_K = tgt_K.unsqueeze(0)
    elif src_K is not None:
        tgt_K = src_K.clone()
    else:
        tgt_K = None

    src_crop = TF.crop(src_resized, top, left, crop_size, crop_size)

    transform_pixels = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    src_tensor = transform_pixels(src_crop).to(dtype=torch.float16, device=DEVICE)
    src = src_tensor.unsqueeze(0)
    base_name = splitext(basename(src_path))[0]
    return src, src_K, tgt_K, base_name


def collect_image_paths(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(input_path, ext)))
        paths = sorted(paths)
        if not paths:
            raise FileNotFoundError(f"No images found in directory: {input_path}")
        return paths
    else:
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")
        return [input_path]


def stack_optional(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    """Given a list of either None or (1,...) tensors, return (B,...) or None."""
    if tensors and tensors[0] is not None:
        return torch.cat(tensors, dim=0)
    return None


def generate_novel_view(args, config, stereo_nvs=None):
    K_src = (
        torch.tensor(args.src_intrinsics, dtype=torch.float32).reshape(3, 3).to(DEVICE)
        if args.src_intrinsics
        else None
    )
    K_tgt = (
        torch.tensor(args.tgt_intrinsics, dtype=torch.float32).reshape(3, 3).to(DEVICE)
        if args.tgt_intrinsics
        else None
    )

    os.makedirs(args.output, exist_ok=True)

    image_paths = collect_image_paths(args.input)
    crop_size = int(config.data.train_width)

    if stereo_nvs is None:
        stereo_nvs = StereoSpace(config, DEVICE)

    bs = max(1, int(args.batch_size))
    for start in range(0, len(image_paths), bs):
        batch_paths = image_paths[start : start + bs]

        src_list = []
        intr_src_list = []
        intr_tgt_list = []
        names = []

        for p in batch_paths:
            src_1, intr_src_1, intr_tgt_1, base_name = process_image(
                p, K_src, K_tgt, crop_size=crop_size
            )
            src_list.append(src_1)
            intr_src_list.append(intr_src_1)
            intr_tgt_list.append(intr_tgt_1)
            names.append(base_name)

        src_batch = torch.cat(src_list, dim=0)  # (B, C, H, W)
        intr_src_batch = stack_optional(intr_src_list)  # (B,3,3) or None
        intr_tgt_batch = stack_optional(intr_tgt_list)  # (B,3,3) or None

        # Baseline: same value for all in the batch. Shape (B,1) to match many stereo APIs.
        baseline = torch.full(
            (src_batch.size(0), 1),
            float(args.baseline),
            device=DEVICE,
            dtype=torch.float32,
        )

        with autocast(DEVICE):
            synthesized = stereo_nvs.perform_nvs(
                src_batch,
                baseline,
                intrinsics=intr_src_batch,
                intrinsics_tgt=intr_tgt_batch,
            )  # expect (B, C, H, W) in [0,1] or similar

        for i, name in enumerate(names):
            out_syn = synthesized[i].detach().float().cpu()
            syn_pil = TF.to_pil_image(out_syn)
            syn_pil.save(
                join(args.output, f"{name}_generated_{args.baseline:0.2f}.png")
            )

            src_img = ((src_batch[i].detach().float().cpu() + 1) * 0.5).clamp(0, 1)
            src_pil = TF.to_pil_image(src_img)
            src_pil.save(join(args.output, f"{name}_source.png"))

            # Create side-by-side image
            width, height = src_pil.size
            sbs_image = Image.new("RGB", (width * 2, height))
            sbs_image.paste(src_pil, (0, 0))
            sbs_image.paste(syn_pil, (width, 0))
            sbs_path = join(args.output, f"{name}_sbs.png")
            sbs_image.save(sbs_path)

            # Create anaglyph image (red-cyan)
            left_array = np.array(src_pil)
            right_array = np.array(syn_pil)
            anaglyph_array = np.zeros_like(left_array)
            # Red channel from left, green and blue from right
            anaglyph_array[:, :, 0] = left_array[:, :, 0]  # Red from left
            anaglyph_array[:, :, 1] = right_array[:, :, 1]  # Green from right
            anaglyph_array[:, :, 2] = right_array[:, :, 2]  # Blue from right
            anaglyph_image = Image.fromarray(anaglyph_array)
            anaglyph_path = join(args.output, f"{name}_anaglyph.png")
            anaglyph_image.save(anaglyph_path)

        print(
            f"Processed {start+len(batch_paths)}/{len(image_paths)} images", flush=True
        )


def main():
    parser = argparse.ArgumentParser(description="Generate novel view from input image")
    parser.add_argument(
        "--input",
        type=str,
        default="./example_images",
        help="Path to input image or directory",
    )
    parser.add_argument("--output", default="./outputs", help="Output directory")
    parser.add_argument("--config", type=str, default="./configs/stereospace.yaml")
    parser.add_argument(
        "--baseline", type=float, default=0.15, help="Baseline for stereo setup"
    )
    parser.add_argument(
        "--src_intrinsics",
        type=float,
        nargs=9,
        help="Source camera intrinsics: 9 floats (row-major 3x3)",
    )
    parser.add_argument(
        "--tgt_intrinsics",
        type=float,
        nargs=9,
        help="Target camera intrinsics: 9 floats (row-major 3x3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing multiple images",
    )
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")

    generate_novel_view(args, config)


if __name__ == "__main__":
    main()
