import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import FAISS

# Create Sidebar
with st.sidebar:
    st.title("LLM Dialogue App 😊")
    st.markdown('''
    ## About
    This app is designed to chat with your personal pdf articles and books.
    Tools used:
    - [Streamlit](https://streamlit.io/)
    - [Langchain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/)
    ''')
    add_vertical_space(5)
    st.write('❤️ Chat with PDFs ❤️')


def main():
    st.header("💬Hello LLMs🗨️")

    load_dotenv()

    # Take pdf file as input
    pdf_file = st.file_uploader(
        "Please upload your pdf file", type='pdf')
    # TODO- Take 'url' user input

    def get_text_embeddings(text, model="text-embedding-ada-002"):
        print("")

    # Read pdf
    if pdf_file is not None:
        st.write(pdf_file.name)
        pdf_reader = PdfReader(pdf_file)
        # st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)

        # compute & store Embeddings
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        store_name = pdf_file.name[:-4]
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vectorstore, f)


if __name__ == "__main__":
    main()
