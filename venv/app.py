import streamlit as st
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from htmlTemplate import css , bot_template, user_template
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text  += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter =CharacterTextSplitter(
        separator ="\n",
        chunk_size = 1000,
        chunk_overlap =200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks



def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name ="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore



def get_conversation_chain(vectorstore):

    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl",model_kwargs={"temperature":0.5,"max_length":512}
                         )

    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(ques):
    response = st.session_state.conversation({"question":ques}
                                             )
    st.session_state.chat_history = response['chat_history']
    for i, msg in enumerate(st.session_state.chat_history):
        if i %2 ==0:
            st.write(user_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",msg.content),unsafe_allow_html=True)

    st.write(response)


def main():
    load_dotenv()

    st.set_page_config(page_title= "Chat to get research paper information",page_icon=":books:")

    st.write(css,unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PDFs Chat Bot: Ask anything from PDF :books:")
    ques = st.text_input("Ask a question from your documents")
    if ques:
        handle_user_input(ques)

    st.write(user_template.replace("{{MSG}}","HELLO PDF BOT !!"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","HELLO THERE !!"), unsafe_allow_html=True)


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDF here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Preprocessing"):
                #get the pdf texts
                raw_text = get_pdf_text(pdf_docs)

                #get the text chunks
                text_chunks = get_text_chunks(raw_text)

                #create vector store
                vectorstore = get_vectorstore(text_chunks)
                #conversation chain 
                st.session_state.conversation  = get_conversation_chain(vectorstore)
    st.session_state.conversation

if __name__ == '__main__':
    main()
