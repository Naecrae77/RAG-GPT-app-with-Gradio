# RAG-GPT App with Gradio

A lightweight Retrieval-Augmented Generation (RAG) chatbot app built with Gradio, OpenAI, LangChain, and Chroma DB. Easily run, interact, and extend your own AI-powered assistant with vector search and custom knowledge.

## Features
- Gradio web interface for chat
- OpenAI integration for responses
- LangChain for vector search and document handling
- Chroma DB for fast, local vector storage and retrieval
- Simple configuration with YAML and .env

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your OpenAI API key in a `.env` file.
3. (Optional) Prepare your documents for vector search using Chroma DB.
4. Run `utils/RAGGPT_gradio_app.py` to start the Gradio web interface::
   ```bash
   python utils/RAGGPT_gradio_app.py
   ```

## Folder Structure
- `utils/` — Core app scripts:
   - `RAGGPT_gradio_app.py` — Main entry point to launch the Gradio app
   - `chatbot.py` — Chatbot logic
   - `vectordb.py` — Chroma DB vector database integration
   - `load_config.py` — YAML config loader
- `requirements.txt` — Major dependencies

## Chroma DB Usage
Chroma DB is used for efficient local vector storage and retrieval. The app automatically manages Chroma DB when you run the main script. You can customize or extend document ingestion in `vectordb.py`.

## Configuration
You can change app settings and parameters in the `config.yaml` file.

## License
MIT
