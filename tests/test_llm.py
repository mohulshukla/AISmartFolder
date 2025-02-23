import pytest
from src.llms.openaillm import OpenAILLM
from src.llms.llama import LlamaLLM
import os


@pytest.fixture
def llm():
    """Fixture to create an OpenAI LLM instance."""
    return LlamaLLM()


def test_openai_llm_basic_response(llm):
    """Test that the OpenAI LLM can generate a basic response."""
    # Test message
    messages = [("user", "What is 2+2?")]

    # Generate response
    response = llm.generate(messages)

    # Check that the response contains a number and is related to the question
    assert response is not None
    assert len(response) > 0
    # Check that the response contains "4" or "four" as this is a basic math question
    assert any(answer in response.lower() for answer in ["4", "four"])
