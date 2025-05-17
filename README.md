# ChatGraph

ChatGraph is a prototype branchable chat application with a visual conversation tree and Retrieval-Augmented Generation (RAG) memory. It uses OpenAI's GPT models and FAISS for semantic search, and is built with Streamlit for an interactive UI.

## Features
- Branchable chat: Fork conversations at any point and explore different paths.
- Visual conversation tree: See your chat history as a branching tree.
- RAG memory: The assistant can reference similar past Q&A pairs using OpenAI embeddings and FAISS.

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/chatgraph.git
cd chatgraph
```

### 2. Install dependencies
We recommend using a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up your `.env` file
Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=sk-...
```

- Your API key is required for both chat and embedding features.
- The app will automatically load this file using `python-dotenv`.

## Running the App

Start the Streamlit app with:
```bash
streamlit run app.py
```

Then open the provided local URL in your browser.

## Notes
- This is a prototype. Some features (like advanced RAG and memory) are under development.
- For questions or issues, please open an issue on GitHub. 