import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil
from src.llms import LLM
from PIL import Image
import mimetypes
from pathlib import Path
from typing import Literal, List, Tuple, Optional
import base64
from mistralai import Mistral
import tempfile
from src.notifications import send_notification


class SmartFolderHandler(FileSystemEventHandler):
    def __init__(self, folder_path: str, llm: LLM):
        self.folder_path = folder_path
        self.llm = llm
        self.mistral_client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

    def get_subfolders(self) -> List[str]:
        """Get list of subfolders in the watched directory."""
        subfolders = []
        for item in os.listdir(self.folder_path):
            item_path = os.path.join(self.folder_path, item)
            if os.path.isdir(item_path):
                subfolders.append(item)
        return subfolders

    def is_image_file(self, file_path: str) -> bool:
        try:
            Image.open(file_path)
            return True
        except:
            return False

    def is_pdf_file(self, file_path: str) -> bool:
        """Check if a file is a PDF."""
        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type == "application/pdf"
        except:
            return False

    def extract_pdf_text(self, file_path: str) -> str:
        """Extract text from a PDF file using Mistral's OCR."""
        try:
            # Upload the PDF file
            with open(file_path, "rb") as f:
                uploaded_pdf = self.mistral_client.files.upload(
                    file={
                        "file_name": os.path.basename(file_path),
                        "content": f,
                    },
                    purpose="ocr",
                )

            # Get signed URL
            signed_url = self.mistral_client.files.get_signed_url(
                file_id=uploaded_pdf.id
            )

            # Get OCR results
            ocr_response = self.mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": signed_url.url,
                },
            )

            # Clean up the uploaded file
            self.mistral_client.files.delete(file_id=uploaded_pdf.id)

            # Extract text from OCR response
            # The response contains pages with markdown content
            text_content = []
            for page in ocr_response.pages:
                if page.markdown:
                    text_content.append(page.markdown)

            return "\n\n".join(text_content)
        except Exception as e:
            print(f"Error extracting PDF text: {str(e)}")
            return ""

    def get_file_content(self, file_path: str) -> tuple[str, bytes]:
        """Get the content of a file."""
        if self.is_image_file(file_path):
            with open(file_path, "rb") as f:
                return "image", f.read()
        elif self.is_pdf_file(file_path):
            text = self.extract_pdf_text(file_path)
            return "text", text.encode("utf-8")
        else:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return "text", f.read().encode("utf-8")
            except:
                return "binary", b""

    def suggest_folder(self, file_path: str) -> str:
        """Suggest the best matching subfolder for the file."""
        file_type, content = self.get_file_content(file_path)
        subfolders = self.get_subfolders()

        if not subfolders:
            print("<thinking>No subfolders exist yet</thinking>")
            print("<answer>none</answer>")
            return ""  # No subfolders exist yet

        # Prepare messages for LLM
        messages: List[Tuple[Literal["user", "assistant", "system"], str]] = [
            (
                "system",
                "You are an AI assistant that helps organize files. Given a file's content and a list of existing folders, "
                "suggest which folder would be the best semantic match for this file. Structure your response with XML tags: "
                "<thinking> for your analysis and <answer> for the final folder name. You must choose from the following folders: "
                + ", ".join(subfolders)
                + "\n\n"
                "Example response:\n"
                "<thinking>This appears to be a Python script with machine learning code, using TensorFlow and neural networks</thinking>\n"
                "<answer>ml_projects</answer>\n\n",
            ),
            (
                "user",
                f"Please analyze this {file_type} and suggest the best matching folder.",
            ),
        ]

        # Get LLM suggestion
        if file_type == "image":
            try:
                response = self.llm.generate(
                    [(role, msg) for role, msg in messages], image=content
                )
            except Exception as e:
                print(f"<thinking>Error getting LLM suggestion: {str(e)}</thinking>")
                print("<answer>none</answer>")
                return ""
        else:
            try:
                response = self.llm.generate(
                    [(role, msg) for role, msg in messages]
                    + [("user", content.decode("utf-8"))]
                )
            except Exception as e:
                print(f"<thinking>Error getting LLM suggestion: {str(e)}</thinking>")
                print("<answer>none</answer>")
                return ""

        if not response:
            print("<thinking>Received empty response from LLM</thinking>")
            print("<answer>none</answer>")
            return ""

        print(response.strip())  # This will print the XML-formatted response

        # Extract just the answer part (between <answer> tags)
        import re

        try:
            answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            suggested_folder = (
                answer_match.group(1).strip().lower() if answer_match else "none"
            )
        except Exception as e:
            print(f"<thinking>Error parsing LLM response: {str(e)}</thinking>")
            print("<answer>none</answer>")
            return ""

        # Only return the suggestion if it matches an existing folder
        if suggested_folder in [f.lower() for f in subfolders]:
            return next(f for f in subfolders if f.lower() == suggested_folder)
        return ""

    def suggest_name(self, file_path: str) -> str:
        """Suggest a descriptive name for the file."""
        file_type, content = self.get_file_content(file_path)
        original_name = os.path.basename(file_path)

        if file_type == "binary":
            print("<thinking>File is binary, keeping original name</thinking>")
            print(f"<answer>{original_name}</answer>")
            return original_name

        # Prepare messages for LLM
        messages: List[Tuple[Literal["user", "assistant", "system"], str]] = [
            (
                "system",
                "You are an AI assistant that helps name files descriptively. Given a file's content, "
                "suggest a clear and descriptive name (without extension). Structure your response with XML tags: "
                "<thinking> for your analysis and <answer> for the final name.\n\n"
                "Example responses:\n"
                "1. <thinking>This Python script implements a neural network for image classification using TensorFlow</thinking>\n"
                "<answer>image_classification_neural_net</answer>\n\n"
                "2. <thinking>This is a quarterly financial report for Q3 2023 showing revenue and expenses</thinking>\n"
                "<answer>q3_2023_financial_report</answer>\n\n"
                "3. <thinking>This image shows a landscape photo of mountains during sunset</thinking>\n"
                "<answer>mountain_sunset_landscape</answer>",
            ),
            (
                "user",
                f"Please analyze this {file_type} and suggest a descriptive name.",
            ),
        ]

        # Get LLM suggestion
        if file_type == "image":
            try:
                response = self.llm.generate(
                    [(role, msg) for role, msg in messages], image=content
                )
            except Exception as e:
                print(f"<thinking>Error getting LLM suggestion: {str(e)}</thinking>")
                print(f"<answer>{original_name}</answer>")
                return original_name
        else:
            try:
                response = self.llm.generate(
                    [(role, msg) for role, msg in messages]
                    + [("user", content.decode("utf-8"))]
                )
            except Exception as e:
                print(f"<thinking>Error getting LLM suggestion: {str(e)}</thinking>")
                print(f"<answer>{original_name}</answer>")
                return original_name

        if not response:
            print("<thinking>Received empty response from LLM</thinking>")
            print(f"<answer>{original_name}</answer>")
            return original_name

        print(response.strip())  # This will print the XML-formatted response

        # Extract just the answer part (between <answer> tags)
        import re

        try:
            answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
            suggested_name = (
                answer_match.group(1).strip().replace(" ", "_")
                if answer_match
                else original_name
            )
        except Exception as e:
            print(f"<thinking>Error parsing LLM response: {str(e)}</thinking>")
            print(f"<answer>{original_name}</answer>")
            return original_name

        return suggested_name

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = str(event.src_path)
        file_name = os.path.basename(file_path)

        # Wait a brief moment to ensure the file is fully written
        time.sleep(1)

        try:
            # Get AI suggestions
            suggested_name = self.suggest_name(file_path)
            suggested_folder = self.suggest_folder(file_path)

            # Keep extension from original file
            _, ext = os.path.splitext(file_path)
            new_name = f"{suggested_name}{ext}"

            # Determine target directory
            if suggested_folder:
                target_dir = os.path.join(self.folder_path, suggested_folder)
            else:
                target_dir = self.folder_path

            # Create full target path
            new_path = os.path.join(target_dir, new_name)

            # Ensure unique filename
            counter = 1
            while os.path.exists(new_path):
                new_name = f"{suggested_name}_{counter}{ext}"
                new_path = os.path.join(target_dir, new_name)
                counter += 1

            # Move the file
            if target_dir != os.path.dirname(file_path):
                shutil.move(str(file_path), new_path)
                relative_path = os.path.relpath(new_path, self.folder_path)
                print(f"Moved {file_name} to {relative_path}")

                # Show notification for moved file
                send_notification(
                    title="📁 File Organized",
                    message=f"Successfully organized {file_name}",
                    subtitle=f"Moved to {suggested_folder}",
                    sound=True,
                )
            else:
                # If staying in same directory, only rename if name changed
                if new_name != file_name:
                    shutil.move(str(file_path), new_path)
                    print(f"Renamed {file_name} to {new_name}")

                    # Show notification for renamed file
                    send_notification(
                        title="✏️ File Renamed",
                        message=f"Old name: {file_name}",
                        subtitle=f"New name: {new_name}",
                        sound=True,
                    )

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            # Show error notification
            send_notification(
                title="❌ Error Processing File",
                message=f"Could not process {file_name}",
                subtitle=str(e),
                sound=True,
            )


def start_smart_folder(folder_path: str, llm: LLM):
    event_handler = SmartFolderHandler(folder_path, llm)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
