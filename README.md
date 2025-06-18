# PDF RAG with Ollama

## Introduction

This project demonstrates Retrieval-Augmented Generation (RAG) over PDF documents using local Large Language Models (LLMs) and embedding models powered by Ollama. It enables you to ask questions about the content of your PDF files and receive intelligent, context-aware answers. The workflow leverages LangChain, ChromaDB, and Ollama for both embeddings and LLM inference.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pranav271103/ollama-pdf-rag
   cd ollama-pdf-rag
   ```

2. **Install Python dependencies:**
   Run these commands in your Jupyter notebook or terminal:
   ```python
   %pip install --quiet unstructured langchain
   %pip install --quiet "unstructured[all-docs]"
   %pip install --quiet chromadb langchain-text-splitters
   ```
   If you encounter issues with ONNX or unstructured, follow the troubleshooting steps in the notebook to reinstall them.

3. **Install and run Ollama:**
   - Download and install Ollama from [https://ollama.ai/](https://ollama.ai/)
   - Start the Ollama service on your machine

4. **Pull the required Ollama models:**
   - For embeddings:
     ```bash
     ollama pull nomic-embed-text
     ```
   - For LLM inference (example):
     ```bash
     ollama pull granite3-dense:2b
     ```
   - You may use other supported models as needed. See [Ollama's model library](https://ollama.ai/library) for more options.

## Usage

1. Place your PDF document(s) in the project directory.
2. Open and run the `RAG.ipynb` notebook:
   ```bash
   jupyter notebook RAG.ipynb
   ```
3. Follow the notebook steps:
   - Install any missing dependencies as prompted.
   - Specify the path to your PDF file.
   - The notebook will load the PDF, split it into chunks, and create vector embeddings using the `nomic-embed-text` model from Ollama.
   - The embeddings are stored in a Chroma vector database.
   - The notebook sets up a retrieval-augmented generation (RAG) chain using a local LLM (e.g., `granite3-dense:2b` from Ollama).
   - You can then ask questions about your PDF, and the system will answer using the retrieved context.

## Notes
- If you encounter dependency issues, refer to the troubleshooting steps in the notebook (e.g., reinstalling ONNX, unstructured, or creating a new environment).
- Ensure that the required Ollama models are pulled before running the relevant cells.
- For best results, keep your Python and Jupyter environments up to date.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

For more details, see comments and instructions within the `RAG.ipynb` notebook. 
