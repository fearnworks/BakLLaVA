# image_processing.py
import numpy as np
import cv2
from PIL import Image, ImageOps
from typing import Optional, Tuple, List, Dict

def get_bounding_box(img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Calculate the bounding box of non-zero pixels in an image.

    Args:
        img (np.ndarray): A numpy array representing the image.

    Returns:
        Optional[Tuple[int, int, int, int]]: The bounding box (min_x, min_y, max_x, max_y),
                                             or None if the image is empty.
    """
    if not np.any(img):  # protect against an empty img
        return None
    non_zero_indices = np.nonzero(img)
    min_x, max_x = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    min_y, max_y = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    return (min_x, min_y, max_x, max_y)

def composite_all_layers(base: np.ndarray, objects: List[Dict]) -> np.ndarray:
    """
    Composite multiple image layers into a single image.

    Args:
        base (np.ndarray): The base image array.
        objects (List[Dict]): List of objects, where each object contains an 'img' key with its image array.

    Returns:
        np.ndarray: The composited image.
    """
    img = base.copy()
    for obj in objects:
        for i in range(obj['img'].shape[0]):
            for j in range(obj['img'].shape[1]):
                if obj['img'][i, j, 3] != 0:
                    img[i, j] = obj['img'][i, j]
    return img

def get_base_layer_mask(state: Dict) -> Tuple[Image.Image, Dict]:
    """
    Generate a mask for the base layer of segmented objects.

    Args:
        state (Dict): The state dictionary containing the segmented image and changed objects.

    Returns:
        Tuple[Image.Image, Dict]: The masked image and the updated state.
    """
    changed_obj_id = [obj['id'] for obj in state['changed_objects']]
    img, mask = state['orignal_segmented'], np.zeros(img.shape[:2], dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 3] in changed_obj_id:
                mask[i, j] = 255
    state['base_layer_mask'] = mask
    mask_image = Image.fromarray(mask)
    mask_image = mask_image.convert("L") if mask_image.mode != "L" else mask_image
    mask_image = ImageOps.invert(mask_image)
    orig_image = Image.fromarray(img[:, :, :3])
    masked_image = Image.composite(orig_image, Image.new(orig_image.mode, orig_image.size, (0, 0, 0, 0)), mask_image)
    return masked_image, state

def get_enlarged_masked_background(state: Dict, mask_dilate_slider: int) -> Tuple[Image.Image, Dict]:
    """
    Get the masked background image with an enlarged mask.

    Args:
        state (Dict): The state dictionary containing the segmented image and mask.
        mask_dilate_slider (int): The dilation level for the mask.

    Returns:
        Tuple[Image.Image, Dict]: The masked image with enlarged background and the updated state.
    """
    mask = state['base_layer_mask']
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_dilate_slider, mask_dilate_slider))
    mask_dilated = cv2.dilate(mask, kernel)
    mask_image = Image.fromarray(mask_dilated)
    mask_image = mask_image.convert("L") if mask_image.mode != "L" else mask_image
    mask_image = ImageOps.invert(mask_image)
    state['base_layer_mask_enlarged'] = mask_image
    img = state['orignal_segmented']
    orig_image = Image.fromarray(img[:, :, :3])
    transparent = Image.new(orig_image.mode, orig_image.size, (0, 0, 0, 0))
    masked_image = Image.composite(orig_image, transparent, mask_image)
    return masked_image, state
