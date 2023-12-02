from abc import ABC, abstractmethod

class BaseComponent(ABC):
    """
    A base class for Gradio UI components to establish a common interface and shared methods.
    """

    def __init__(self):
        """
        Initialize the base component.
        """
        super().__init__()
        self.component = None
        self._create_ui_elements()

    @abstractmethod
    def _create_ui_elements(self):
        """
        Abstract method for creating UI elements specific to the component.
        """
        pass

    @abstractmethod
    def setup_event_handlers(self):
        """
        Abstract method for setting up event handlers for the component.
        """
        pass

    def get_component(self):
        """
        Get the main UI element of the component.

        Returns:
            The Gradio UI element for the component.
        """
        return self.component