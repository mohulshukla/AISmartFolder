import os
import pytest
import shutil
from PIL import Image
import io
from src.smart_folder import SmartFolderHandler
from src.llms.llama import LlamaLLM


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    test_dir = tmp_path / "test_smart_folder"
    test_dir.mkdir()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.fixture
def llm():
    """Create a mock LLM instance for testing."""
    return LlamaLLM()


@pytest.fixture
def handler(temp_dir, llm):
    """Create a SmartFolderHandler instance with a temporary directory."""
    return SmartFolderHandler(str(temp_dir), llm)


@pytest.fixture
def sample_folders(temp_dir):
    """Create sample subfolders for testing."""
    folders = [
        "photos_of_people",
        "financial_docs",
        "python_code",
        "music_sheets",
        "scanned_documents",
        "meeting_notes",
    ]
    for folder in folders:
        (temp_dir / folder).mkdir()
    return folders


def create_test_image(path, color="red"):
    """Create a small test image."""
    img = Image.new("RGB", (100, 100), color=color)
    img.save(path)


def test_get_subfolders(handler, sample_folders):
    """Test getting list of subfolders."""
    subfolders = handler.get_subfolders()
    assert sorted(subfolders) == sorted(sample_folders)


def test_is_image_file(handler, temp_dir):
    """Test image file detection."""
    # Test with image file
    image_path = temp_dir / "test.png"
    create_test_image(image_path)
    assert handler.is_image_file(str(image_path)) is True

    # Test with text file
    text_path = temp_dir / "test.txt"
    text_path.write_text("Hello, World!")
    assert handler.is_image_file(str(text_path)) is False


def test_get_file_content(handler, temp_dir):
    """Test getting file content for different file types."""
    # Test text file
    text_path = temp_dir / "test.txt"
    text_path.write_text("Hello, World!")
    content_type, content = handler.get_file_content(str(text_path))
    assert content_type == "text"
    assert content.decode("utf-8") == "Hello, World!"

    # Test image file
    image_path = temp_dir / "test.png"
    create_test_image(image_path)
    content_type, content = handler.get_file_content(str(image_path))
    assert content_type == "image"
    assert len(content) > 0


def test_suggest_folder_for_documents(handler, sample_folders):
    """Test folder suggestion for different types of documents."""
    # Test financial document
    invoice_path = handler.folder_path + "/restaurant_bill.txt"
    with open(invoice_path, "w") as f:
        f.write(
            """INVOICE #2024-0315
TOTAL AMOUNT DUE: $113.08
PAYMENT DETAILS:
Credit Card: VISA ****4321
Transaction ID: TX-789012
Date: March 15, 2024

ITEMIZED BILL:
1. Appetizer - $12.99
2. Main Course - $24.99
3. Dessert - $8.99
4. Beverages - $15.99

Subtotal: $62.96
Tax (8.5%): $5.35
Tip (20%): $12.59
Total: $80.90

Thank you for your business!
Please keep this receipt for your records."""
        )

    suggestion = handler.suggest_folder(invoice_path)
    assert suggestion.lower() == "financial_docs"  # Very clearly a financial document

    # Test Python code file
    code_path = handler.folder_path + "/data_processor.py"
    with open(code_path, "w") as f:
        f.write(
            """#!/usr/bin/env python3
import pandas as pd
import numpy as np

def process_data(data_frame):
    '''Process pandas DataFrame and return analytics'''
    return {
        'mean': data_frame.mean(),
        'median': data_frame.median(),
        'std_dev': data_frame.std()
    }

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    results = process_data(df)
    print(results)"""
        )

    suggestion = handler.suggest_folder(code_path)
    assert suggestion.lower() == "python_code"  # Clearly Python code

    # Test scanned document
    doc_path = handler.folder_path + "/scanned_contract.txt"
    with open(doc_path, "w") as f:
        f.write(
            """SCANNED DOCUMENT
[Document ID: SCAN-2024-03-15-001]
[Scanned at: 300 DPI]
[Scanner: HP OfficeJet Pro]
[Time: 14:30:45]

EMPLOYMENT CONTRACT
-----------------
This agreement is made between..."""
        )

    suggestion = handler.suggest_folder(doc_path)
    assert suggestion.lower() == "scanned_documents"  # Clearly a scanned document


def test_suggest_folder_for_images(handler, sample_folders):
    """Test folder suggestion for different types of images."""
    # Use real test images from tests/images
    image_path = "tests/images/music.png"  # Scanned document

    # Test image categorization
    suggestion = handler.suggest_folder(image_path)
    assert suggestion.lower() == "music_sheets"  # Since it's a scanned document


def test_suggest_name(handler):
    """Test name suggestion for different types of files."""
    # Test naming a Python data processing script
    code_path = handler.folder_path + "/script.py"
    with open(code_path, "w") as f:
        f.write(
            """#!/usr/bin/env python3
import pandas as pd

def analyze_sales_data(csv_path):
    '''Analyze monthly sales data and generate report'''
    df = pd.read_csv(csv_path)
    monthly_totals = df.groupby('month')['sales'].sum()
    return monthly_totals

if __name__ == '__main__':
    results = analyze_sales_data('sales.csv')
    print('Monthly Sales Analysis:', results)"""
        )

    name = handler.suggest_name(code_path)
    assert "sales" in name.lower() and "analysis" in name.lower()

    # Test naming a financial document
    invoice_path = handler.folder_path + "/document.txt"
    with open(invoice_path, "w") as f:
        f.write(
            """MARCH 2024 ELECTRICITY BILL
Account: #12345-67
Customer: John Smith
Billing Period: March 1-31, 2024
Amount Due: $157.23
Due Date: April 15, 2024

Usage Details:
kWh Used: 750
Rate: $0.21/kWh"""
        )

    name = handler.suggest_name(invoice_path)
    assert (
        "march" in name.lower()
        and "electricity" in name.lower()
        and "bill" in name.lower()
    )


def test_file_organization(handler, sample_folders, temp_dir):
    """Test complete file organization process with real content."""
    # Create test files
    files = [
        (
            "invoice_march.txt",
            """INVOICE #123
Amount: $500
Date: March 15, 2024""",
        ),
        (
            "hello_world.py",
            """def main():
    print("Hello, World!")""",
        ),
        (
            "meeting_notes.txt",
            """DevOps Team Meeting Notes
Date: March 15, 2024
Attendees: John, Jane, Bob""",
        ),
    ]

    # Process each file
    for filename, content in files:
        file_path = temp_dir / filename
        with open(file_path, "w") as f:
            f.write(content)

        event = type("Event", (), {"is_directory": False, "src_path": str(file_path)})()
        handler.on_created(event)

    # Verify files were organized correctly
    assert len(list((temp_dir / "financial_docs").glob("*invoice*.txt"))) == 1
    assert len(list((temp_dir / "python_code").glob("*.py"))) == 1
    assert len(list((temp_dir / "meeting_notes").glob("*dev*.txt"))) == 1


def test_duplicate_handling(handler, sample_folders, temp_dir):
    """Test handling of duplicate files with real content."""
    # Create two similar files
    content = """DevOps Team Meeting Notes
Date: March 15, 2024
Topic: Project Updates"""

    for i in range(2):
        file_path = temp_dir / f"meeting_{i}.txt"
        with open(file_path, "w") as f:
            f.write(content)

        event = type("Event", (), {"is_directory": False, "src_path": str(file_path)})()
        handler.on_created(event)

    # Verify files were renamed uniquely
    doc_files = list((temp_dir / "meeting_notes").glob("*dev*.txt"))
    assert len(doc_files) == 2
    assert doc_files[0].name != doc_files[1].name


def test_error_handling(handler):
    """Test error handling for invalid files."""
    event = type(
        "Event", (), {"is_directory": False, "src_path": "/nonexistent/file.txt"}
    )()
    handler.on_created(event)  # Should not raise exception
