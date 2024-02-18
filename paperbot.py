import os
import tempfile # PDFアップロードの際に必要

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import base64
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image

# ローカル
# import sqlite3 

# 本番環境
from langchain.text_splitter import CharacterTextSplitter
import translate
import openai

import sqlite3
# import pysqlite3 as sqlite3
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


openai.api_key = st.secrets.OpenAIAPI.openai_api_key
os.environ["OPENAI_API_KEY"] = st.secrets.OpenAIAPI.openai_api_key

st.set_page_config(layout="wide")

def show_pdf(file_path:str):
    """Show the PDF in Streamlit
    That returns as html component

    Parameters
    ----------
    file_path : [str]
        Uploaded PDF file path
    """

    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000" type="application/pdf">'
    st.markdown(pdf_display, unsafe_allow_html=True)

folder_name = "./.data"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# ストリーム表示
class StreamCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens_area = st.empty()
        self.tokens_stream = ""

    def on_llm_new_token(self, token, **kwargs):
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

# UI周り
st.title("ろんぶんらぐちゃっと")

tab_chat, tab_search, tab_pdf, tab_data = st.tabs(["CHAT", "SEARCH", "PDF UPLOAD","DATA"])

with st.sidebar:
    # 画像を表示
    img = Image.open('paperrobotpic.jpg')
    st.image(img, use_column_width=True)

    st.markdown("## モデルの設定")
    select_model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct", "gpt-4-1106-preview", "gpt-4-turbo-preview"])
    
    st.markdown("## PDF読み込みパラメータの設定")
    select_temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1,)
    select_chunk_size = st.slider("Chunk", min_value=0.0, max_value=1000.0, value=512.0, step=10.0,)
    select_overlap = st.slider("Chunk-overlap", min_value=0.0, max_value=select_chunk_size, value=0.0, step=10.0,)

    st.markdown("## 検索パラメータの設定")
    select_fetch_k = st.slider("MMR使用件数 fetch_k", min_value=1, max_value=50, value=20, step=1)
    select_k = st.slider("検索数 k", min_value=1, max_value=select_fetch_k, value=5, step=1)
    select_lambda = st.slider("検索結果多様性 λ", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    translate_on = st.toggle('検索結果を翻訳する', key='1')

with tab_pdf:
    uploaded_file = st.file_uploader("PDFをアップロードしてください", type="pdf")
    if uploaded_file:
        # 一時ファイルにPDFを書き込みバスを取得
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
            # PDFを表示
            # with st.expander("読み込まれたファイル", expanded=False):
            #     st.write(show_pdf(tmp_file.name))
            #     #st.write(uploaded_file.getvalue())
            loader = PyMuPDFLoader(file_path=tmp_file_path) 
            documents = loader.load() 
            
            # text_splitter = RecursiveCharacterTextSplitter(
            #     chunk_size = 500,
            #     chunk_overlap  = 128,
            #     length_function = len,
            # )
            
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                separator="."
                ,chunk_size=select_chunk_size
                ,chunk_overlap=select_overlap
            )
            
            # documentsから改行コード\nを削除する
            for doc in documents:
                doc.page_content = doc.page_content.replace("\n", "")

            data = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
            )

            database = Chroma(
                # persist_directory="./.data",
                embedding_function=embeddings,
            )
            database.add_documents(data)
        tmp_file.close()

with tab_chat:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
    )
    database = Chroma(
        # persist_directory="./.data",
        embedding_function=embeddings,
    )
    chat = ChatOpenAI(
        model=select_model,
        temperature=select_temperature,
        streaming=True,
    )

    # retrieverに変換（検索、プロンプトの構築）
    retriever = database.as_retriever(search_type="mmr", search_kwargs={'fetch_k': select_fetch_k, "k": select_k, 'lambda_mult': select_lambda})

    # 会話履歴を初期化
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

    memory = st.session_state.memory

    chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )

    # UI用の会話履歴を初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # UI用の会話履歴を表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # UI周り
    container = st.container(border=True)
    prompt = st.chat_input("PDFに関する質問を入力してください", key='unique_key_1')
    if prompt:
        # UI用の会話履歴に追加
        st.session_state.messages.append({"role": "user", "content": prompt})
        with container.chat_message("user"):
            st.markdown(prompt)

        with container.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chain(
                    {"question": prompt},
                    callbacks=[StreamCallbackHandler()], # ストリーム表示
                )
                
                #st.markdown(response["answer"])

                # 回答のソースドキュメントを表示
                source_documents = response["source_documents"]
                for souece_no, source_document in enumerate(source_documents):
                    with st.expander(f"Source{souece_no}: {source_document.metadata['title']}(p{source_document.metadata['page'] + 1})", expanded=False):
                        #　テキストを表示
                        st.write(source_document.page_content)
                        
                        # 画像を表示
                        # page_number = source_document.metadata['page']  # 表示したいページ番号
                        # documents_fitz = fitz.open(source_document.metadata['file_path'])
                        # page = documents_fitz.load_page(page_number)  # 0-indexed
                        # pix = page.get_pixmap()
                        # img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        # st.image(img, caption=f'Page {page_number+1}', use_column_width=True)
                        
                        # メタデータを表示
                        df = pd.DataFrame(source_document.metadata, index=["ソースに関する情報"])
                        st.table(df.transpose())

            # UI用の会話履歴に追加
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    
    # メモリの内容をターミナルで確認
    print(memory)
with tab_search:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
    )
    database = Chroma(
        # persist_directory="./.data",
        embedding_function=embeddings,
    )
    search_message = st.chat_input("PDFに関する質問を入力してください", key='unique_key_2')
    if search_message:
        retriever = database.as_retriever(search_type="mmr", search_kwargs={'fetch_k': select_fetch_k, "k": select_k, 'lambda_mult': select_lambda})
        docs = retriever.get_relevant_documents(search_message)
        st.markdown(f"##  \"{search_message}\" の検索結果")
        for num, source_document in enumerate(docs):
            st.markdown("---")
            #　テキストを表示
            # search_messageに一致する単語をハイライト
            st.markdown(f"### 検索結果{num+1}")
            st.markdown(f"- 論文名: {source_document.metadata['title']} / p.{source_document.metadata['page'] + 1}")            

            st.markdown(f"#### 検索結果テキスト")
            seached_txt = source_document.page_content.replace(search_message, f"**`{search_message}`**")
            st.markdown(seached_txt)

            # テキストを日本語に翻訳して表示
            if translate_on:
                seached_txt_jp = translate.get_translation(source_document.page_content)
                st.markdown(f"#### 検索結果テキスト日本語訳")
                st.markdown(seached_txt_jp)
    
            with st.expander("該当PDFページ", expanded=False):    
                # 画像を表示
                page_number = source_document.metadata['page']  # 表示したいページ番号
                documents_fitz = fitz.open(source_document.metadata['file_path'])
                page = documents_fitz.load_page(page_number)  # 0-indexed
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                st.image(img, caption=f'Page {page_number+1}', use_column_width=False)
                
                # メタデータを表示
                df = pd.DataFrame(source_document.metadata, index=["ソースに関する情報"])
                st.table(df.transpose())
# chromaのデータをGUIで確認
with tab_data:
    # database = Chroma(
    #     # persist_directory="./.data",
    #     embedding_function=embeddings,
    # )
    # # データの確認
    # conn = sqlite3.connect("./chroma.sqlite3")

    # # テーブル名を取得
    # cursor = conn.cursor()
    # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    
    # # embedding_metadataテーブルの情報を取得
    # cursor.execute("PRAGMA table_info(collection_metadata);")

    # # 結果を取得
    # tables = cursor.fetchall()
    # # Streamlitで結果を表示
    # for table in tables:
    #     st.write(table)

    # ボタンを押すとchromaのデータをすべて削除
    if st.button("データを削除"):
        # chromaのすべてのレコードのIDを取得
        conn = sqlite3.connect("./chroma.sqlite3")
        c = conn.cursor()

        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = c.fetchall()

        # 各テーブルを削除
        for table in tables:
            c.execute(f'TRUNCATE {table[0]};')

        # 変更をコミット
        conn.commit()