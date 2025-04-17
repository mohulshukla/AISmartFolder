from src.llms import LLM
from typing import List, Tuple, Literal, Optional, Union
import os
import base64
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient

load_dotenv()


class LlamaStackLLM(LLM):
    """
    Llama implementation of the LLM interface using llama-stack.
    Supports both text-only and vision-enabled models.
    """

    def __init__(self):
        """
        Initialize the Llama LLM.

        Args:
            api_key: API key (not used for local llama-stack)
        """
        self.client = LlamaStackClient(base_url="http://localhost:8321")
        models = self.client.models.list()
        self.model_id = next(m for m in models if m.model_type == "llm").identifier

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
        Generate a response using llama-stack API.

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

        response = self.client.inference.chat_completion(
            model_id=self.model_id,
            messages=formatted_messages,
        )

        content = str(response.completion_message.content)
        if content is None:
            raise ValueError("Received empty response from llama-stack API")
        return content
