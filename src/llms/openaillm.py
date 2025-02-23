from src.llms import LLM
from openai import OpenAI
from typing import List, Tuple, Literal, Optional, Union
import os
import base64


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

        response = self.client.chat.completions.create(
            model=self.model, messages=formatted_messages, max_tokens=2000
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Received empty response from OpenAI API")
        return content
