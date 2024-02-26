import openai
import streamlit as st
import pandas as pd
from transformers import GPT2Tokenizer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import os

# Configurações do ChatBot
st.title("Este é o ChatBot desenvolvido por Pedro Sampaio Amorim. Inclua um texto para debater com o bot!")

openai.api_key = st.secrets['OPENAI_API_KEY']

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Verifica se um arquivo foi enviado
arquivo_upload = st.file_uploader("Escolha um arquivo TXT", type=["txt"])
if arquivo_upload is not None:
    st.success("Arquivo enviado com sucesso!")

    # Lê o conteúdo do arquivo enviado
    conteudo_bytes = arquivo_upload.read()
    conteudo = conteudo_bytes.decode('utf-8')  # Converte bytes para string

    # Recebe a entrada do usuário do arquivo enviado (Tipo 1)
    prompt_tipo_1 = conteudo
    st.session_state.messages.append({"role": "user", "content": prompt_tipo_1, "tipo": "tipo_1"})

# Exibe o histórico de mensagens
for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])
    elif message["role"] == "user":
        if message.get("tipo") == "tipo_1":
            # Mensagens do Tipo 1 (contexto) não são exibidas na tela
            continue
        with st.chat_message("user_tipo_2"):
            st.markdown(f"**Usuário:** {message['content']}")

# Recebe a entrada do usuário (Tipo 2)
if prompt_tipo_2 := st.text_input("Enviou o texto ? Se sim, o que você gostaria de discutir sobre ele? Caso não queira falar sobre texto, do que deseja falar?"):
    st.session_state.messages.append({"role": "user", "content": prompt_tipo_2, "tipo": "tipo_2"})

    # Gera a resposta do ChatBot
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
