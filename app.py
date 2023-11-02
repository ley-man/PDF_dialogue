import os
import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from dotenv import find_dotenv, load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback

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
        # st.write(pdf_file.name)
        pdf_reader = PdfReader(pdf_file)
        # st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

        chunks = text_splitter.split_text(text=text)

        store_name = pdf_file.name[:-4]

        if os.path.exists(path=f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            st.write(f":green[Embeddings Loaded from disk]")
        else:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorstore, f)
            st.write(f":red[Embeddings Computed]")

        # User query
        query = st.text_input(
            f":blue[Please ask questions about your document.]")
        # st.write(query)
        if query:
            # TODO: Incorporate llm response along with doc resopnse
            docs = vectorstore.similarity_search(query=query, k=2)
            # st.write(docs)

            # llm = OpenAI(model_name='gpt-3.5-turbo', temperature=0.)
            llm = OpenAI(temperature=0, model_name="text-davinci-003")

            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:

                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == "__main__":
    main()
