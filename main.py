from src.smart_folder import start_smart_folder
from src.llms.openaillm import OpenAILLM
from src.llms.llama import LlamaLLM
import os

if __name__ == "__main__":
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "folder")
    print(f"Starting Smart Folder system. Watching: {folder_path}")
    print("Press Ctrl+C to stop")
    llm = LlamaLLM()
    start_smart_folder(folder_path, llm)
