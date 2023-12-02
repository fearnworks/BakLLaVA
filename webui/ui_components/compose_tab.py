from typing import Any
from webui.ui_components.base_component import BaseComponent
import webui.image.segment_generation as SegGen
import webui.image.image_processing as ImgProc
import webui.state_management as StateMgmt
import gradio as gr

class ComposeTabComponent(BaseComponent):
    def __init__(self, sketch_pad) -> None:
        """
        Initialize the ComposeTabComponent with the Gradio interface module.
        """
        self._create_ui_elements()
        self.sketch_pad = sketch_pad

    def _create_ui_elements(self) -> None:
        """
        Create UI elements specific to the Compose Tab.
        """
        self.compose_tab = gr.Tab("Remove or Change Objects")
        with self.compose_tab:
            gr.Markdown(
                "Segment an object by drawing a stroke or giving a referring text. "
                "Then press the segment button. Drag the highlighted object to move it. "
                "To remove it, drag it out of the frame. To replace it with a new object, "
                "give an instruction only if the object is removed and press the generate "
                "button until you like the image."
            )
            with gr.Row().style(equal_height=False):
                with gr.Column():
                    with gr.Group():
                        with gr.Column():
                            with gr.Row():
                                self.segment_task = gr.Radio(
                                    ["Stroke", "Text"], value="Stroke", label='Choose segmentation method'
                                )
                                self.segment_text = gr.Textbox(label="Enter referring text")
                            self.segment_btn = gr.Button("Segment", elem_id="segment-btn")

                    with gr.Group():
                        self.segmented_img = gr.Image(label="Move or delete object", tool="compose", height=256)

                    with gr.Group():
                        with gr.Column():
                            grounding_text_box = gr.Textbox(
                                label="Enter grounding text for generating a new image"
                            )
                            with gr.Row():
                                self.compose_clear_btn = gr.Button("Clear", elem_id="compose_clear_btn")
                                self.compose_btn = gr.Button("Generate", elem_id="compose_btn")

                    with gr.Accordion("Advanced Options", open=False):
                        with gr.Row():
                            self.masked_background_img = gr.Image(
                                label="Background", type='pil', interactive=False, height=256
                            )
                            self.inpainted_background_img = gr.Image(
                                label="Inpainted Background", type='pil', interactive=False, height=256
                            )
                        self.mask_dilate_slider = gr.Slider(
                            minimum=0.0,
                            maximum=100,
                            value=50,
                            step=2,
                            interactive=True,
                            label="Mask dilation",
                            visible=True,
                            scale=20,
                        )
                        with gr.Row(visible=False):
                            self.compose_fix_seed = gr.Checkbox(value=False, label="Fixed seed", visible=False)
                            self.compose_rand_seed = gr.Slider(
                                minimum=0, maximum=1000, step=1, value=0, label="Seed", visible=False
                            )

    def get_component(self) -> Any:
        """
        Return the main UI element of the Compose Tab component.
        
        Returns:
            Any: The Gradio UI element for the Compose Tab.
        """
        return self.compose_tab
    
    def setup_event_listeners(self, compose_state=StateMgmt.compose_state, shared_state=StateMgmt.shared_state):
        self.segment_btn.click(
            self.handle_segmentation,
            inputs=[self.segment_task, self.segment_text, self.mask_dilate_slider, compose_state],
            outputs=[self.segmented_img, compose_state]
        )

        self.compose_btn.click(
            self.handle_generation,
            inputs=[self.grounding_text_box, compose_state],
            outputs=[self.segmented_img, compose_state]
        )

        self.compose_clear_btn.click(StateMgmt.load_shared_state, [shared_state], self.sketch_pad)
        
    def handle_segmentation(self, segment_task, segment_text, mask_dilate_slider, compose_state=StateMgmt.compose_state):
        img = {'image': self.current_image, 'mask': self.current_mask}  # Assuming these are defined
        segmented_image, enlarged_background, base_layer_inpainted, updated_state = SegGen.get_segments(
            img, segment_task, segment_text, mask_dilate_slider, compose_state
        )
        # Update UI elements and state as needed
        self.segmented_img.update(value=segmented_image)
        compose_state.update(updated_state)
        return segmented_image, compose_state
        
    def handle_generation(self, grounding_text_box, compose_state=StateMgmt.compose_state):
        """
        Generate new content based on user inputs.

        Args:
            grounding_text_box: The text input for grounding the generation.
            compose_state: The state object for the compose tab.
        """
        # Assuming grounding_text_box contains the necessary text for generation
        grounding_text = grounding_text_box
        generated_image, updated_state = ImgGen.get_generated(
            grounding_text, compose_state['fix_seed'], compose_state['rand_seed'], compose_state
        )

        # Update the UI with the generated image
        self.segmented_img.update(value=generated_image)
        compose_state.update(updated_state)
        return generated_image, compose_state
