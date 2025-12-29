# Automated Insurance Claims Validator (Multi-Modal RAG)

This repository contains a specialized Multi-Modal Retrieval Augmented Generation (RAG) pipeline designed for the insurance industry. It automates the forensic analysis of insurance claims by processing mixed-media evidence folders containing both official PDF reports and raw image evidence (JPG/PNG).

The system utilizes parallel processing and a tiered Large Language Model (LLM) architecture to ingest data efficiently, assess damage severity, and cross-reference visual evidence against written claimant statements to detect potential inconsistencies or fraud.

## Key Features

* **Multi-Modal Ingestion:** Simultaneously processes unstructured text (PDFs, forms) and visual data (accident photos, receipts) from a unified directory.
* **Forensic Analysis:** Uses custom prompting to act as a claims adjuster, analyzing images for specific damage patterns (e.g., paint transfer, rust) rather than generic descriptions.
* **Parallel Processing:** Implements Python's `concurrent.futures` to ingest and analyze files in parallel, significantly reducing processing time for large claim folders.
* **Cost-Optimized Architecture:** Utilizes a tiered model strategy:
* **GPT-4o-mini** for high-volume indexing and initial image analysis.
* **GPT-4o** for the final complex reasoning and verdict generation.


* **Smart Image Compression:** Automatically resizes and compresses high-resolution evidence photos before API transmission to reduce token usage and latency.
* **Evidence Linking:** Retrieves and cites specific image evidence in the final output, allowing for verifiable decision-making.

## System Architecture

1. **Data Loading:** The system scans a target directory for `.pdf`, `.jpg`, `.jpeg`, and `.png` files.
2. **Preprocessing:**
* PDFs are partitioned into text chunks and tables.
* Images are resized and compressed.


3. **Indexing (Parallel):** Files are processed in parallel threads. Visual data is sent to the vision model for damage assessment; text data is summarized.
4. **Vector Storage:** Analyzed chunks are embedded and stored in a local ChromaDB instance.
5. **Retrieval & Reasoning:** When queried, the system retrieves relevant text and images, de-duplicates evidence, and uses a high-intelligence model to generate a final validation verdict.

## Prerequisites

* Python 3.8 or higher
* OpenAI API Key (with access to GPT-4o and GPT-4o-mini)
* **System Dependencies:** This project relies on `unstructured`, which requires specific system-level tools for PDF and image processing.

## Installation

### 1. Install System Dependencies

You must install these tools before installing the Python libraries.

**For Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr libmagic-dev

```

**For macOS (using Homebrew):**

```bash
brew install poppler tesseract libmagic

```

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/insurance-claims-rag.git
cd insurance-claims-rag

```

### 3. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

```

### 4. Install Python Packages

Create a `requirements.txt` file with the following content (or run the command below):

```text
langchain
langchain-community
langchain-openai
unstructured[all-docs]
chromadb
pillow
openai
tiktoken

```

Then install them:

```bash
pip install -r requirements.txt

```

## Configuration

Set your OpenAI API key as an environment variable. You can do this in your terminal or by creating a `.env` file.

**Terminal Method:**

```bash
export OPENAI_API_KEY="sk-..."

```

**Python Script Method:**
Add this to the top of your main script if not using environment variables:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

```

## Usage

### 1. Prepare Data

Create a directory (e.g., `claims_data/`) and drop your mixed media files into it.

* Example: `claims_data/claim_12345/`
* `police_report.pdf`
* `bumper_photo.jpg`
* `repair_estimate.pdf`



### 2. Run the Pipeline

Save the optimized code provided in the project documentation as `main.py`.

Modify the execution block at the bottom of `main.py` to point to your data folder:

```python
if __name__ == "__main__":
    # Initialize and Ingest
    # This will create a local 'db_claims_optimized' folder
    vector_db = parallel_ingest_folder("./claims_data/claim_12345")

    # Run a query
    query = "Is the visual damage on the bumper consistent with the description in the police report?"
    response = ask_expert_adjuster(query, vector_db)
    
    print("Final Verdict:")
    print(response)

```

### 3. Execute

```bash
python main.py

```

## Troubleshooting

* **Tesseract Not Found:** If you receive an error regarding Tesseract, ensure it is installed and added to your system PATH.
* **OpenAI Rate Limits:** If processing a very large folder, you may hit rate limits. Adjust the `max_workers` in the `parallel_ingest_folder` function to a lower number (e.g., 2 or 3).
* **ChromaDB Errors:** If you encounter database errors, try deleting the generated `db_claims_optimized` folder to force a clean re-indexing.

## License

This project is open-source and is licensed under the MIT License.
