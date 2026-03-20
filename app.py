import os
print("\n".join(f"{k}={v}" for k, v in os.environ.items()))
os.system("pip freeze")

import glob
import gradio as gr
import shutil
import spaces
import subprocess
import sys
import tempfile
from PIL import Image
from gradio_patches.radio import Radio

REPO_URL = "https://github.com/prs-eth/stereospace.git"
COMMIT_SHA = "d7bbae6"
REPO_DIR = "stereospace"
DEVICE = "cuda"



def clone_repository():
    if os.path.exists(REPO_DIR) and os.path.isdir(os.path.join(REPO_DIR, ".git")):
        print(f"Repository {REPO_DIR} already exists, checking out commit...")
        subprocess.run(["git", "fetch"], cwd=REPO_DIR, check=True, capture_output=True)
        subprocess.run(["git", "checkout", COMMIT_SHA], cwd=REPO_DIR, check=True)
    else:
        print(f"Cloning repository {REPO_URL} at commit {COMMIT_SHA}...")
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
        subprocess.run(["git", "checkout", COMMIT_SHA], cwd=REPO_DIR, check=True)
    print(f"Repository ready at {REPO_DIR}")


#clone_repository()

sys.path.insert(0, REPO_DIR)
from inference import generate_novel_view
from src import StereoSpace



def create_placeholder_image():
    placeholder = Image.new('RGB', (1, 1), color='black')
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        placeholder.save(tmp.name)
        return tmp.name


def find_output_file(output_dir, base_name, output_mode):
    if output_mode == "Anaglyph":
        pattern = os.path.join(output_dir, f"{base_name}_anaglyph.png")
    elif output_mode == "Side-by-side":
        pattern = os.path.join(output_dir, f"{base_name}_sbs.png")
    elif output_mode == "Generated view":
        pattern = os.path.join(output_dir, f"{base_name}_generated_*.png")
        matches = glob.glob(pattern)
        if matches:
            pattern = matches[0]
        else:
            raise FileNotFoundError(f"No generated file found matching {pattern}")
    elif output_mode == "Input view":
        pattern = os.path.join(output_dir, f"{base_name}_source.png")
    else:
        raise ValueError(f"Unknown output mode: {output_mode}")
    
    if not os.path.exists(pattern):
        raise FileNotFoundError(f"Output file not found: {pattern}")
    
    return pattern


def find_all_output_files(output_dir, base_name):
    outputs = {}
    modes = ["Anaglyph", "Side-by-side", "Generated view", "Input view"]
    for mode in modes:
        try:
            output_file = find_output_file(output_dir, base_name, mode)
            outputs[mode] = output_file
        except FileNotFoundError:
            pass
    return outputs


@spaces.GPU
def process_all_modes(input_image):
    if input_image is None:
        raise gr.Error("Please upload an image or select an example.")
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_input:
        input_path = tmp_input.name
        if isinstance(input_image, str):
            shutil.copy(input_image, input_path)
        else:
            Image.open(input_image).convert("RGB").save(input_path)
    
    try:
        with tempfile.TemporaryDirectory() as tmp_output:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            inference_script = "inference.py"
            cmd = [
                "python", inference_script,
                "--input", input_path,
                "--output", tmp_output
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            print(f"Working directory: {REPO_DIR}")
            result = subprocess.run(
                cmd,
                cwd=REPO_DIR,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                error_msg = f"Inference failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                print(error_msg)
                raise gr.Error(error_msg)
            
            try:
                output_files = find_all_output_files(tmp_output, base_name)
            except Exception as e:
                all_files = os.listdir(tmp_output)
                error_msg = f"Output files not found. Available files: {all_files}\nError: {str(e)}"
                print(error_msg)
                raise gr.Error(error_msg)
            
            outputs = {}
            for mode, output_file in output_files.items():
                output_image = Image.open(output_file).convert("RGB")
                output_image.load()
                outputs[mode] = output_image
            
            return outputs
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)

def get_example_images():
    example_dir = os.path.join(REPO_DIR, "example_images")
    if not os.path.exists(example_dir):
        return []
    
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    examples = []
    for ext in image_extensions:
        examples.extend(glob.glob(os.path.join(example_dir, ext)))
    
    return sorted(examples)


with gr.Blocks(
    title="StereoSpace Demo",
) as demo:
    gr.Markdown(
        """
        <div align="center">
            <h2>StereoSpace: Depth-Free Synthesis of Stereo Geometry via End-to-End Diffusion in a Canonical Space</h2>
        </div>
        """
    )
    with gr.Row(elem_classes="remove-elements"):
        gr.Markdown(
            f"""
            <p align="center">
            <a title="Website" href="https://hf.co/spaces/prs-eth/stereospace_web" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/badge/%E2%99%A5%20Project%20-Website-blue">
            </a>
            <a title="arXiv" href="https://arxiv.org/abs/2512.10959" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/badge/%F0%9F%93%84%20arXiv%20-Paper-AF3436">
            </a>
            <a title="Github" href="https://github.com/prs-eth/stereospace" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/github/stars/prs-eth/stereospace?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
            </a>
            <a title="Model weights" href="https://hf.co/prs-eth/stereospace-v1-0" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model%20-Weights-yellow" alt="imagedepth">
            </a>
            <a title="Social" href="https://twitter.com/antonobukhov1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                <img src="https://shields.io/twitter/follow/:?label=Subscribe%20for%20updates!" alt="social">
            </a>
            </p>                    
            <p align="center" style="margin-top: 0px;">
                Upload a photo or pick an example below to create stereo space, wait for the result, then watch it in anaglyph, side-by-side, or generated view.
                If a quota limit appears, duplicate the space to continue.
            </p>
            """
    )

    with gr.Row():
        output_mode = Radio(
            choices=["Anaglyph", "Side-by-side", "Input view", "Generated view"],
            value="Anaglyph",
            label=None,
            container=False,
            scale=1,
            elem_classes="horizontal-radio"
        )
    
    with gr.Row():
        image = gr.Image(
            type="filepath",
            label="Input/Output Image",
            elem_classes="result-image",
            height=480,
        )
    
    outputs_gallery = gr.Gallery(
        visible=False,
        label="Computed Outputs",
        show_label=False,
        value=[],
    )
    
    def update_image_from_gallery(gallery_data, current_mode, current_image=None):
        if gallery_data is None or not gallery_data:
            return current_image if current_image is not None else None
        
        result = None
        for item in gallery_data:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                img, label = item[0], item[1]
                if label == current_mode:
                    result = img
                    break
            elif isinstance(item, str):
                continue
        
        if result is None and gallery_data:
            first_item = gallery_data[0]
            if isinstance(first_item, (list, tuple)) and len(first_item) >= 1:
                result = first_item[0]
            elif isinstance(first_item, str):
                result = first_item
        
        if result is None:
            return current_image if current_image is not None else None
        
        if isinstance(result, Image.Image):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                result.save(tmp.name)
                return tmp.name
        
        return result
    
    def process_new_image(img, current_mode):
        if img is None:
            placeholder = create_placeholder_image()
            return [], placeholder, gr.update()
        
        all_outputs = process_all_modes(img)        
        gallery_data = []
        available_modes = []
        for mode_name, mode_image in all_outputs.items():
            gallery_data.append([mode_image, mode_name])
            available_modes.append(mode_name)
        
        placeholder = create_placeholder_image()
        radio_value = current_mode if current_mode in available_modes else (available_modes[0] if available_modes else "Anaglyph")
        return gallery_data, placeholder, gr.update(choices=available_modes, value=radio_value)
    
    def process_example_simple(img):
        if img is None:
            placeholder = create_placeholder_image()
            return [], placeholder, gr.update()
        
        all_outputs = process_all_modes(img)
        
        gallery_data = []
        available_modes = []
        for mode_name, mode_image in all_outputs.items():
            gallery_data.append([mode_image, mode_name])
            available_modes.append(mode_name)
        
        placeholder = create_placeholder_image()
        radio_value = available_modes[0] if available_modes else "Anaglyph"
        return gallery_data, placeholder, gr.update(choices=available_modes, value=radio_value)
    
    def clear_image():
        return None, []
    
    examples_list = get_example_images()
    if examples_list:
        def process_example_wrapper(img):
            gallery_data, placeholder_image, radio_update = process_example_simple(img)
            return gallery_data, placeholder_image, radio_update
        
        examples_component = gr.Examples(
            examples=examples_list,
            inputs=[image],
            outputs=[outputs_gallery, image, output_mode],
            fn=process_example_wrapper,
            cache_examples=True,
            cache_mode="lazy",
            label="Example Images",
            elem_id="example-images-gallery",
        )
    
    def process_upload_wrapper(img, current_mode):
        gallery_data, blocked_image, radio_update = process_new_image(img, current_mode)
        return gallery_data, blocked_image, radio_update
    
    upload_event = image.upload(
        fn=process_upload_wrapper,
        inputs=[image, output_mode],
        outputs=[outputs_gallery, image, output_mode]
    )
    
    def on_gallery_change(gallery_data, current_mode, current_image):
        if not gallery_data or len(gallery_data) == 0:
            return current_image, gr.update(interactive=True)
        updated_image = update_image_from_gallery(gallery_data, current_mode, current_image)
        return updated_image, gr.update(interactive=True)
    
    gallery_change_event = outputs_gallery.change(
        fn=on_gallery_change,
        inputs=[outputs_gallery, output_mode, image],
        outputs=[image, image]
    )
    
    def switch_mode_handler(current_mode, gallery_data, current_image):
        updated_image = update_image_from_gallery(gallery_data, current_mode, current_image)
        return updated_image
    
    output_mode.change(
        fn=switch_mode_handler,
        inputs=[output_mode, outputs_gallery, image],
        outputs=image
    )
    
    image.clear(
        fn=clear_image,
        outputs=[image, outputs_gallery]
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        ssr_mode=False,
        css="""
            #example-images-gallery button[class*="gallery-item"][class*="svelte-"] {
                min-width: max(96px, calc(100vw / 8));
                min-height: max(96px, calc(100vw / 8));
                width: max(96px, calc(100vw / 8));
                height: max(96px, calc(100vw / 8));
            }
            #example-images-gallery button[class*="gallery-item"] div[class*="container"] {
                min-width: max(96px, calc(100vw / 8));
                min-height: max(96px, calc(100vw / 8));
                width: max(96px, calc(100vw / 8));
                height: max(96px, calc(100vw / 8));
            }
            #example-images-gallery button[class*="gallery-item"] img {
                min-width: max(96px, calc(100vw / 8));
                min-height: max(96px, calc(100vw / 8));
                width: max(96px, calc(100vw / 8));
                height: max(96px, calc(100vw / 8));
                object-fit: cover;
            }
        """,
        head="""
            <script>
                let observerFooterButtons = new MutationObserver((mutationsList, observer) => {
                    const origButtonShowAPI = document.querySelector(".show-api");
                    const origButtonBuiltWith = document.querySelector(".built-with");
                    const origButtonSettings = document.querySelector(".settings");
                    const origSeparatorDiv = document.querySelector(".divider");
                    if (!origButtonBuiltWith || !origButtonShowAPI || !origButtonSettings || !origSeparatorDiv) {
                        return;
                    }
                    observer.disconnect();
                    const parentDiv = origButtonShowAPI.parentNode;
                    if (!parentDiv) return;
                    const createButton = (referenceButton, text, href) => {
                        let newButton = referenceButton.cloneNode(true);
                        newButton.href = href;
                        newButton.textContent = text;
                        newButton.className = referenceButton.className;
                        newButton.style.textDecoration = "none";
                        newButton.style.display = "inline-block";
                        newButton.style.cursor = "pointer";
                        return newButton;
                    };
                    document.querySelectorAll(".divider").forEach(divider => {
                        divider.style.marginLeft = "var(--size-2)";
                        divider.style.marginRight = "var(--size-2)";
                    });
                    
                    const newButtonBuiltWith = createButton(origButtonBuiltWith, "Built with Gradio DualVision", "https://github.com/toshas/gradio-dualvision");
                    const newButtonTemplateBy = createButton(origButtonBuiltWith, "Template by Anton Obukhov", "https://www.obukhov.ai");
                    const newButtonLicensed = createButton(origButtonBuiltWith, "Licensed under CC BY-SA 4.0", "http://creativecommons.org/licenses/by-sa/4.0/");
                    parentDiv.replaceChild(newButtonBuiltWith, origButtonShowAPI);
                    parentDiv.replaceChild(newButtonTemplateBy, origButtonBuiltWith);
                    parentDiv.replaceChild(newButtonLicensed, origButtonSettings);
                });
                observerFooterButtons.observe(document.body, { childList: true, subtree: true });
            </script>
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-1FWSVCGZTG"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag() {{dataLayer.push(arguments);}}
                gtag('js', new Date());
                gtag('config', 'G-1FWSVCGZTG');
            </script>
        """
    )