"""
Workaround for https://github.com/gradio-app/gradio/issues/12564
"""
import gradio as gr


class Radio(gr.Radio):
    # Default values for attributes that Block.get_config() and Component.get_config() expect
    _default_attributes = {
        # Block attributes (from Block.__init__)
        'proxy_url': None,
        'rendered_in': None,
        'key': None,
        'visible': True,
        'elem_id': None,
        'elem_classes': [],
        'parent': None,
        'is_rendered': False,
        # Component attributes (from Component.__init__)
        'info': None,
        'server_fns': [],
        '_selectable': False,
        'label': None,
    }
    
    def __getattr__(self, name):
        if name not in self._default_attributes:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        value = self._default_attributes[name]
        setattr(self, name, value)
        return value
