import gradio as gr
from typing import Dict, Union

class ImageMask(gr.components.Image):
    """
    Custom Gradio Image component with specialized preprocessing for masking.

    Inherits from gr.components.Image and overwrites its preprocessing method.
    """

    is_template = True

    def __init__(self, **kwargs):
        """
        Initialize the ImageMask component.

        Args:
            **kwargs: Keyword arguments for the Gradio Image component.
        """
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x: Union[str, Dict, None]) -> Union[str, Dict, None]:
        """
        Custom preprocessing for the ImageMask component.

        Args:
            x (Union[str, Dict, None]): The input to be preprocessed.

        Returns:
            Union[str, Dict, None]: The preprocessed input.
        """
        if isinstance(x, str):
            x = {'image': x, 'mask': x}
        elif isinstance(x, dict):
            if x['mask'] is None and x['image'] is None:
                x
            elif x['image'] is None:
                x['image'] = str(x['mask'])
            elif x['mask'] is None:
                x['mask'] = str(x['image'])
        elif x is not None:
            assert False, 'Unexpected type {0} in ImageMask preprocess()'.format(type(x))
        return super().preprocess(x)
