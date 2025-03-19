# DocuQuery: AI-Powered PDF Knowledge Assistant

## Overview
DocuQuery is a Streamlit-based web application that lets users upload PDF documents and ask questions about their content. The app extracts text from the PDFs, splits it into manageable chunks, and generates embeddings using Google Generative AI via LangChain. These embeddings are stored in a FAISS vector store for efficient retrieval, and a conversational retrieval chain is set up to answer user questions interactively.

## Features
* **PDF Text Extraction**: Extracts text from one or more PDF files.
* **Text Splitting**: Splits extracted text into smaller chunks for improved processing.
* **Embeddings Generation**: Uses GoogleGenerativeAIEmbeddings to generate embeddings from text.
* **Vector Store**: Stores and retrieves embeddings efficiently using FAISS.
* **Conversational Q&A**: Sets up a conversational retrieval chain with a supported Google Gemini model for interactive querying.

## Installation
1. **Clone the Repository**:
   ```
   git clone git@github.com:mahitha-07/DocQuery---Nasscom.git
   cd DocuQuery-Nasscom
   ```

2. **Create and Activate a Virtual Environment**:
   ```
   python -m venv venv
   ```
   * For Windows: `venv\Scripts\activate`
   * For macOS/Linux: `source venv/bin/activate`

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Setup
1. **Create a `.env` file** with your Google API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```
   The application uses the python-dotenv package to load these variables.

2. **Ensure All Dependencies Are Installed**:
   Make sure your requirements.txt includes:
   * streamlit
   * PyPDF2
   * langchain
   * langchain-google-genai
   * faiss-cpu
   * langchain-community
   * python-dotenv

## Usage
1. **Run the Application**:
   ```
   streamlit run app.py
   ```

2. **Interact with the App**:
   * **Upload Documents**: Use the sidebar to upload one or more PDF files.
   * **Process Documents**: Click the "Process" button to extract text, generate embeddings, and build the vector store.
   * **Ask Questions**: Enter your question in the main text input field and view the conversation history with responses generated by the model.

## Dependencies
* **Streamlit**: For building the interactive UI.
* **PyPDF2**: For extracting text from PDF files.
* **LangChain**: For text splitting and building the conversational retrieval chain.
* **langchain-google-genai**: For Google Generative AI integrations.
* **FAISS-cpu**: For vector storage and similarity search.
* **python-dotenv**: For loading environment variables from a .env file.

## License
This project is licensed under the MIT License.

## Acknowledgments
* Google Gen AI SDK Documentation [https://googleapis.github.io/python-genai/](https://googleapis.github.io/python-genai/)
* LangChain Documentation [https://python.langchain.com/](https://python.langchain.com/)
* FAISS – Facebook AI Similarity Search [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
