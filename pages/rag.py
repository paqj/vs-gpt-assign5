from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory

import os
from tempfile import gettempdir
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

# callback handler
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# 세션 상태에서 API 키 가져오기
if 'api_key' in st.session_state and st.session_state['api_key']:
    user_api_key = st.session_state['api_key']
    # API 키가 있는 경우 ChatOpenAI 객체에 적용
    llm = ChatOpenAI(
        api_key=user_api_key,  # API 키 적용
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )
    # Memory
    memory = ConversationBufferMemory(
        llm=llm,
        max_token_limit=50,
        return_messages=True,
    )
else:
    # API 키 입력을 요청하는 메시지
    st.title("API Key is not set. Please enter your API key on app page")
    with st.sidebar:
        st.error("API Key is not set. Please enter your API key on app page")
    # `llm` 객체와 관련된 코드 실행을 중지
    st.stop()

# Memory
memory = ConversationBufferMemory(
    llm=llm,
    max_token_limit=50,
    return_messages=True,
)

# 임시 디렉토리 경로 사용 예시
temp_dir = gettempdir()

# 동일한 file(hashing)이면 구동되지 않고, 직전에 실행된 결과를 리턴.
@st.cache_data(show_spinner="Embdding file...")
def embed_file(file, user_api_key):
    file_content = file.read()
    
    # Caching File
    file_path = os.path.join(temp_dir, file.name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(file_content)

    # Caching Embeddings
    cache_base_dir = os.path.join(temp_dir, "streamlit_cache", "embeddings")
    file_cache_dir = os.path.join(cache_base_dir, file.name)
    os.makedirs(file_cache_dir, exist_ok=True)
    cache_dir = LocalFileStore(file_cache_dir)

    # splitter
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    # embedding
    embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

# Uploader를 sidebar로 이동
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

    st.markdown(
        """
        [GitHub Repository](https://github.com/paqj/vs-gpt-assgin5)
        """,
    )

if file and user_api_key:
    retriever = embed_file(file, user_api_key)

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

    message = st.chat_input("Ask Anything about your file...")

    if message:
        send_message(message, "human")
        # User의 Input에서 retriver를 호출 -> 추출한 docs를 -> prompt에 전달(context, question) -> llm
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(), # "question": message
            }
            | prompt
            | llm
        )

        # ai의 답변으로 보이게 함
        with st.chat_message("ai"):
            response = chain.invoke(message)


# file이 없으면, history 초기화
else:
    st.session_state["messages"] = []