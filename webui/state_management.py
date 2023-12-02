import gradio as gr 

def save_shared_state(img, state):
    """
    Save the current image to the shared state.
    """
    if isinstance(img, dict) and 'image' in img:
        state['working_image'] = img['image']
    else:
        state['working_image'] = img
    return state

def load_shared_state(state, task=None):
    """
    Load the shared state based on the task.
    """
    if task == "Grounded Generation":
        return None
    else:
        return state['working_image']

def update_shared_state(state, task):
    """
    Update the shared state based on the current task.
    """
    if task == "Grounded Generation":
        state['working_image'] = None
    return state

def clear_grounding_info(state):
    """
    Clear grounding information from the state.
    """
    state['boxes'] = []
    state['masks'] = []
    return state, ''

compose_state = gr.State(
            {
                'boxes': [],
                'move_no': 0,
                'base_layer': None,
                'segment_info': None,
                'seg_boxes': {},
                'changed_objects': [],
            }
        )

llava_state = gr.State()
shared_state = gr.State({'working_image': None})