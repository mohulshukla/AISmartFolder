from src.llms import LLM
from openai import OpenAI
from typing import List, Tuple, Literal, Optional, Union
import os
import base64
from dotenv import load_dotenv

load_dotenv()


class LlamaLLM(LLM):
    """
    Llama implementation of the LLM interface using OpenRouter.
    Supports both text-only and vision-enabled models.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Llama LLM.

        Args:
            api_key: OpenRouter API key. If None, will try to get from OPENROUTER_API_KEY environment variable
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key must be provided either through constructor or OPENROUTER_API_KEY environment variable"
            )

        self.client = OpenAI(
            api_key=self.api_key, base_url="https://openrouter.ai/api/v1"
        )
        self.model = "meta-llama/llama-3.2-11b-vision-instruct"

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
        Generate a response using Llama via OpenRouter API.

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
            model=self.model,
            messages=formatted_messages,
            max_tokens=2000,
        )
        print(response)

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Received empty response from OpenRouter API")
        return content
