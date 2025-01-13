import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import OpenAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import OpenAI, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv

load_dotenv()
os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("OPENAI_API_KEY"))

# Load the FAISS index
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Initialize empty chat history
chat_history = []

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    # Initialize the language model
    model = ChatOpenAI(model="gpt-4", temperature=0.3)

    # Initialize the retriever from the loaded FAISS index
    retriever = new_db.as_retriever()

    # Create the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(model, retriever)
    return chain



def user_input(user_question):
    global chat_history  # Use global chat history
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    input_dict = {
        "question": user_question,
        "chat_history": chat_history,
    }

    response = chain.run(input_dict)

    # Update chat history with the new user question and model's response
    chat_history.append(("user", user_question))
    chat_history.append(("assistant", response["output_text"]))

    print(response)
    st.write("Reply: ", response["output_text"])




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using LLM")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
