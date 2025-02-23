from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union


class LLM(ABC):
    """
    Abstract base class representing a Large Language Model (LLM).
    """

    @abstractmethod
    def generate(
        self, messages: List[Tuple[str, str]], image: Optional[Union[str, bytes]] = None
    ) -> str:
        """
        Generate a response based on the input messages and optional image.

        Args:
            messages: List of tuples containing (role, content) pairs.
                     Example: [("user", "Hello"), ("assistant", "Hi there"), ("user", "How are you?")]
            image: Optional image input, can be either a file path (str) or raw bytes

        Returns:
            str: The model's response
        """
        pass
