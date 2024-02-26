import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter  # Importe esta linha
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
import tempfile
import urllib.request
import requests



def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Como posso te ajudar?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Olá, sou seu assistente."]

def conversation_chat(query, chain, history):
    prompt = "Você é um assistente que só conversa no idioma português do Brasil (você nunca, jamais conversa em outro idioma que não seja o português do Brasil):\n\n"  # Adicionando prompt para indicar o idioma
    query_with_prompt = prompt + query
    result = chain({"question": query_with_prompt, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]


def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("question:", placeholder="Me pergunte sobre o(s) conjunto(s) de dados pré-carregados", key='input')
            submit_button = st.form_submit_button(label='Enviar')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

def create_conversational_chain():

    llm = CTransformers(
        streaming=True,
        model="EleutherAI/gpt-3.5-turbo",
        language="pt-BR",
        callbacks=[StreamingStdOutCallbackHandler()],
        input={"temperature": 0.01, "max_length": 500, "top_p": 1}
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff', memory=memory)
    return chain


def main():

    # Initialize session state
    initialize_session_state()
    st.title('[Versão 1.0] 🦙💬 Llama 2 Chatbot.')
    # URL direta para a imagem hospedada no GitHub
    image_url = 'https://raw.githubusercontent.com/pedrosale/falcon_test/6a42509edeb8455594f928c6aa8730d11349f477/fluxo%20atual%202-%20Llama2.jpg'
    # Exibir a imagem usando a URL direta
    st.image(image_url, caption='Arquitetura atual: GitHub + Streamlit')
    st.markdown('**Esta versão contém:**  \nA) Modelo llama2 🦙💬 (llama-2-70b) com refinamento de parâmetros;  \nB) Conjunto de dados pré-carregados do CTB [1. Arquivo de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt) e [2. Reforço de Contexto](https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt);  \nC) Processamento dos dados carregados (em B.) com uso da biblioteca Langchain.')
    # Carrega o arquivo diretamente (substitua o caminho do arquivo conforme necessário)

    # Carrega o primeiro arquivo diretamente
    file_path1 = "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB3.txt"
    with tempfile.NamedTemporaryFile(delete=False) as temp_file1:
        temp_file1.write(urllib.request.urlopen(file_path1).read())
        temp_file_path1 = temp_file1.name

    text1 = []
    loader1 = TextLoader(temp_file_path1)
    text1.extend(loader1.load())
    os.remove(temp_file_path1)
    
    # Carrega o segundo arquivo diretamente
    file_path2 = "https://raw.githubusercontent.com/pedrosale/falcon_test/main/CTB2.txt"
    with tempfile.NamedTemporaryFile(delete=False) as temp_file2:
        temp_file2.write(urllib.request.urlopen(file_path2).read())
        temp_file_path2 = temp_file2.name

    text2 = []
    loader2 = TextLoader(temp_file_path2)
    text2.extend(loader2.load())
    os.remove(temp_file_path2)
    
    # Combina os textos carregados dos dois arquivos
    text = text1 + text2

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=950, length_function=len)
    text_chunks = text_splitter.split_documents(text)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={'device': 'cpu'})

    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

    # Create the chain object
    chain = create_conversational_chain(vector_store)

    display_chat_history(chain)

if __name__ == "__main__":
    main()
