from src.smart_folder import start_smart_folder
from src.llms.openaillm import OpenAILLM
from src.llms.llama import LlamaLLM
from src.llms.llamastack import LlamaStackLLM
import os
import argparse

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Start Smart Folder system with optional custom path"
    )
    parser.add_argument(
        "--path", type=str, help="Path to the folder to watch (optional)"
    )
    args = parser.parse_args()

    # Use provided path or fall back to default
    folder_path = (
        args.path
        if args.path
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "folder")
    )
    print(f"Starting Smart Folder system. Watching: {folder_path}")
    print("Press Ctrl+C to stop")
    llm = LlamaStackLLM()
    start_smart_folder(folder_path, llm)
