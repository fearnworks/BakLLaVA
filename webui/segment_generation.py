import gradio as gr
from typing import Tuple, Dict, Union
from PIL import Image, ImageOps
import numpy as np
import process_image as ImageProcessing
import SEEM as SEEM

def get_segments(img: Dict, task: str, reftxt: str, mask_dilate_slider: int, state: Dict) -> Tuple[Image.Image, np.ndarray, np.ndarray, Dict]:
    """
    Processes the image for segmentation and updates the state with segmented data.

    Args:
        img (Dict): A dictionary containing the image and mask data.
        task (str): The task for segmentation.
        reftxt (str): Referring text for segmentation.
        mask_dilate_slider (int): The current value of the mask dilation slider.
        state (Dict): The current state of the application, containing image data and object states.

    Returns:
        Tuple[Image.Image, np.ndarray, np.ndarray, Dict]: The segmented image, enlarged masked background, base layer inpainted image, and the updated state.
    """
    # Initialize state for segmentation
    state['orignal_segmented'] = None
    state['base_layer'] = None
    state['base_layer_masked'] = None
    state['base_layer_mask'] = None
    state['base_layer_mask_enlarged'] = None
    state['base_layer_inpainted'] = None
    state['segment_info'] = None
    state['seg_boxes'] = {}
    state['changed_objects'] = []
    state['move_no'] = 0

    # Process image for segmentation
    print("Calling SEEM_app.inference")
    # Convert numpy arrays to PIL images
    pil_image = Image.fromarray(img['image']) if isinstance(img['image'], np.ndarray) else None
    pil_mask = Image.fromarray(img['mask']) if isinstance(img['mask'], np.ndarray) else None
    img = {'image': pil_image, 'mask': pil_mask}
    img_ret, seg_info = SEEM.inference(img, task, reftxt=reftxt)
    # Resize to target size
    tgt_size = (img['image'].width, img['image'].height)
    img_ret = img_ret.resize(tgt_size, resample=Image.Resampling.NEAREST)
    state['orignal_segmented'] = np.array(img_ret).copy()
    state['base_layer'] = np.array(img_ret)
    state['segment_info'] = seg_info
    img_ret_array = np.array(img_ret)
    img_ret_array[:, :, 3] = 255 - img_ret_array[:, :, 3]

    # Generate bounding boxes for segmented objects
    for obj_id, label in seg_info.items():
        obj_img = img_ret_array[:, :, 3] == 255 - obj_id
        bbox = ImageProcessing.get_bounding_box(obj_img)
        print(f"obj_id={obj_id}, label={label}, bbox={bbox}")
        state['seg_boxes'][obj_id] = bbox

    # Process the first object as special event
    data = {"index": (0, 0), "value": 254, "selected": True}
    evt = gr.SelectData(None, data)
    mask_dilate_slider, _, state = changed_objects_handler(mask_dilate_slider, state, evt)

    # Update state with masks and inpainted image
    state['base_layer_masked'], state = get_base_layer_mask(state)
    if mask_dilate_slider != 0:
        enlarged_masked_background, state = ImageProcessing.get_enlarged_masked_background(state, mask_dilate_slider)
    state['base_layer_inpainted'] = np.array(ImageProcessing.get_inpainted_background(state, mask_dilate_slider))

    return Image.fromarray(img_ret_array), enlarged_masked_background, state['base_layer_inpainted'], state

def get_base_layer_mask(state: Dict) -> Tuple[Image.Image, Dict]:
    """
    Generates a mask for the base layer of segmented objects and updates the state.

    Args:
        state (Dict): The current state of the application, containing image data and object states.

    Returns:
        Tuple[Image.Image, Dict]: The base layer masked image and the updated state.
    """
    changed_obj_id = [obj['id'] for obj in state['changed_objects']]
    img = state['orignal_segmented']
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 3] in changed_obj_id:
                mask[i, j] = 255
    state['base_layer_mask'] = mask

    mask_image = Image.fromarray(mask)
    if mask_image.mode != "L":
        mask_image = mask_image.convert("L")
    mask_image = ImageOps.invert(mask_image)
    orig_image = Image.fromarray(img[:, :, :3])
    transparent = Image.new(orig_image.mode, orig_image.size, (0, 0, 0, 0))
    masked_image = Image.composite(orig_image, transparent, mask_image)

    return masked_image, state

def changed_objects_handler(mask_dilate_slider: int, state: Dict, evt: gr.SelectData) -> Tuple[int, np.ndarray, Dict]:
    """
    Handles changes in objects based on user interaction.

    Args:
        mask_dilate_slider (int): The current value of the mask dilation slider.
        state (Dict): The current state of the application, containing image data and object states.
        evt (gr.SelectData): Data from Gradio interface about the selection event.

    Returns:
        Tuple[int, np.ndarray, Dict]: Updated mask dilation slider value, the base layer masked image, and the updated state.
    """
    state['move_no'] += 1
    pos_x, pos_y = evt.index  # obj moved out of scene is signaled by (10000, 10000)
    obj_id = 255 - evt.value
    print(f"obj {obj_id} moved by {pos_x}, {pos_y}")
    img = state['base_layer']
    for obj in state['changed_objects']:
        if obj['id'] == obj_id:
            img = obj['img']
            state['changed_objects'].remove(obj)
            break
    new_img = np.zeros_like(img)
    bbox = None
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 3] == obj_id:
                new_i = i + pos_y
                new_j = j + pos_x
                if new_i >= 0 and new_i < img.shape[0] and new_j >= 0 and new_j < img.shape[1]:
                    new_img[new_i, new_j] = img[i, j]
                img[i, j] = 0
    bbox = ImageProcessing.get_bounding_box(new_img)  # returns None if obj moved out of scene
    print("bbox: ", bbox)
    state['changed_objects'].append({'id': obj_id, 'img': new_img, 'text': state['segment_info'][obj_id], 'box': bbox})
    return mask_dilate_slider, state['base_layer_masked'], state

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
