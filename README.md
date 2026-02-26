# Multimodal RAG System

## What it does
This is a comprehensive offline-capable Retrieval-Augmented Generation (RAG) system. It supports semantic search across various modalities including text documents (PDF, DOCX, TXT), images (PNG, JPG, etc.), and audio files (MP3, WAV, etc.). The application features:
- **Text Search**: Answer queries and retrieve context from documents using local embedding models.
- **Image Search**: Uses cross-modal (CLIP) embeddings to search content using an image query.
- **Audio Search**: Transcribes audio using faster-whisper and performs semantic text search on the transcriptions.
- **Local offline processing**: Designed to operate without relying on external cloud APIs by using local GGUF LLMs and embedding models.
- **Streamlit UI**: Offers a beautiful and interactive user interface to query and upload documents.

## Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Allen-Paul-18/LocalLens-OmniRAG.git
   cd LocalLens-OmniRAG
   ```

2. **Set up a Python virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the local models:**
   - Obtain the local LLM model you wish to use (e.g., a GGUF format model like Mistral or Llama).
   - Once downloaded, in `streamlit_app.py`, or in the Streamlit UI, update the path to the model you intend to use.

5. **Run the application:**
   ```bash
   streamlit run streamlit_app.py
   ```
