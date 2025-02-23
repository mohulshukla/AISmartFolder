from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Literal
import base64
from openai import OpenAI
import os


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


class OpenAILLM(LLM):
    """
    OpenAI implementation of the LLM interface.
    Supports both text-only and vision-enabled models.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the OpenAI LLM.

        Args:
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable
            model: Model to use. Defaults to gpt-4-turbo-preview
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either through constructor or OPENAI_API_KEY environment variable"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def _encode_image(self, image: Union[str, bytes]) -> str:
        """Convert image to base64 string."""
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return base64.b64encode(image).decode("utf-8")

    def generate(
        self,
        messages: List[Tuple[Literal["user", "assistant", "system"], str]],
        image: Optional[Union[str, bytes]] = None,
    ) -> str:
        """
        Generate a response using OpenAI's API.

        Args:
            messages: List of (role, content) pairs
            image: Optional image input (file path or bytes)

        Returns:
            The model's response as a string
        """
        formatted_messages = []

        # Format regular messages
        for role, content in messages:
            if image and role == "user":
                # For the user message with image, we need special handling
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self._encode_image(image)}"
                    },
                }
                formatted_messages.append(
                    {
                        "role": role,
                        "content": [{"type": "text", "text": content}, image_content],
                    }
                )
            else:
                formatted_messages.append({"role": role, "content": content})

        # Use vision model if image is provided
        if image:
            self.model = "gpt-4-vision-preview"

        response = self.client.chat.completions.create(
            model=self.model, messages=formatted_messages, max_tokens=2000
        )

        return response.choices[0].message.content
