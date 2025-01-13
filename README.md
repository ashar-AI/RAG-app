# LLM-to-Chat-with-Multiple-PDFs

## End-to-End AI Project to Chat with Multiple PDF Documents using Langchain, OpenAI, and Streamlit

This project is a **Streamlit web application** that allows users to interactively analyze and ask questions based on text extracted from multiple PDF files. The application leverages **OpenAI's GPT models** for natural language processing and uses **Langchain** for managing embeddings, question-answering chains, and interactions with the PDFs.

You can interact with the app live:  
[Streamlit Web App - Chat with PDFs](https://rag-app-9nm3xxfdgagq8bv9rbb8ra.streamlit.app/)

### Project Overview

The web app processes PDF files uploaded by the user, extracts text, generates embeddings using Langchain and OpenAI's GPT models, and sets up a conversational question-answering model. The user can then ask questions based on the contents of the PDFs, and the system responds intelligently, leveraging the extracted text.

### Prerequisites

1. **API Keys**:
   - You need an **OpenAI API key** to use OpenAI's GPT models. This key should be stored in a `.env` file.

2. **Dependencies**:
   - Make sure you have the following Python packages installed:
     - `streamlit`: For creating the web app.
     - `PyPDF2`: For reading PDF files.
     - `langchain`: For managing embeddings and building the question-answering model.
     - `openai`: For interacting with OpenAI's GPT models.
     - `faiss-cpu`: For managing embeddings using FAISS.
     - `langchain-openai`: For OpenAI-specific integrations with Langchain.
     - `python-dotenv`: For loading environment variables from the `.env` file.

### Steps to Get Started

#### 1. **Clone the Repository**:
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/LLM-to-Chat-with-Multiple-PDFs.git
   cd LLM-to-Chat-with-Multiple-PDFs

2. Environment Setup:
The app loads the OpenAI API key from a .env file using load_dotenv().
The openai.api_key = os.getenv("OPENAI_API_KEY") line configures the OpenAI API with your API key.
3. Function Definitions:
get_pdf_text(pdf_docs): Extracts and returns the concatenated text from PDF documents uploaded by the user.
get_text_chunks(text): Splits the extracted text into smaller, manageable chunks.
get_vector_store(text_chunks): Generates embeddings for the text chunks using Langchain and creates a FAISS vector store.
get_conversational_chain(): Sets up the conversational question-answering chain, which uses OpenAI's GPT models.
user_input(user_question): Handles user input by performing a similarity search on the embeddings and generating a response using the conversational chain.
4. Main Function:
The main() function initializes the Streamlit app interface.
The app allows users to upload PDFs, ask questions, and receive responses.
The PDF files are processed, embeddings are generated, and the conversational model is used to answer user queries.
The user interface includes text input fields and a sidebar menu for uploading PDF files and configuring the app.
5. Execution:
The script is executed with if __name__ == "__main__": main() to start the Streamlit app when the script is run directly.
Example Usage
Upload PDF Files: Use the sidebar to upload one or more PDF files.
Ask Questions: Type your question in the text input box and hit Enter to get an answer based on the content of the uploaded PDFs.
Key Features
Multiple PDF Support: The app can handle multiple PDFs at once, combining their text for querying.
Conversational AI: Powered by OpenAI, the app allows natural language conversation with the PDF content.
Fast Query Responses: By using embeddings and FAISS for similarity search, the app provides quick responses to user queries.
Example Workflow
Upload PDFs via the sidebar.
Ask questions such as:
"What is the summary of the document?"
"Can you explain Section 2.1?"
"What are the key points in Chapter 3?"
Get responses based on the content of the uploaded documents.
Troubleshooting
API Key Issues: Ensure your OpenAI API key is correctly set in the .env file.
Missing Dependencies: Make sure all dependencies are installed by running pip install -r requirements.txt.
App Not Loading: Check the Streamlit logs for errors and ensure your environment is correctly set up.