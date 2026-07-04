import streamlit as st
import os
# 设置 Hugging Face 的镜像站
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from langchain_ollama import ChatOllama
# 文本分割器独立包
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 社区文档加载器
from langchain_community.document_loaders import PDFPlumberLoader
# 嵌入模型移到社区包
from langchain_community.embeddings import HuggingFaceEmbeddings
# 向量数据库
from langchain_chroma import Chroma
# QA链
from langchain_classic.chains import RetrievalQA

# ---------------------- 1. 初始化配置 ----------------------
# 向量库持久化路径
CHROMA_PATH = "./chroma_knowledge"
# 本地嵌入模型 bge-small-zh
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh"
# Ollama大模型
LLM_MODEL_NAME = "qwen:7b"

# 加载向量化模型
@st.cache_resource
def get_embedding():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# 加载向量数据库
@st.cache_resource
def get_vector_store():
    embeddings = get_embedding()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    return db

# 加载本地Qwen7B
@st.cache_resource
def get_llm():
    llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.1)
    return llm

# 构建问答RAG链
def build_qa_chain():
    db = get_vector_store()
    llm = get_llm()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# ---------------------- 2. 文档上传与入库 ----------------------
def load_pdf_to_chroma(uploaded_file):
    # 临时保存PDF
    temp_file = f"./temp_{uploaded_file.name}"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.read())
    
    # 加载文档
    loader = PDFPlumberLoader(temp_file)
    docs = loader.load()
    
    # 文本分片
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    split_docs = splitter.split_documents(docs)
    
    # 写入向量库
    db = get_vector_store()
    db.add_documents(split_docs)
    
    # 删除临时文件
    os.remove(temp_file)
    return f"✅ {uploaded_file.name} 知识库导入完成！"

# ---------------------- 3. Streamlit页面UI ----------------------
st.set_page_config(page_title="本地文档问答RAG", layout="wide")
st.title("📖 本地私有知识库问答 | Ollama Qwen7B")

# 侧边栏上传PDF
with st.sidebar:
    st.header("📁 上传PDF文档")
    uploaded_pdf = st.file_uploader("选择PDF文件", type="pdf")
    if uploaded_pdf:
        if st.button("导入知识库"):
            msg = load_pdf_to_chroma(uploaded_pdf)
            st.success(msg)

# 对话输入框
user_query = st.chat_input("请输入你的问题...")
if user_query:
    qa_chain = build_qa_chain()
    with st.spinner("AI正在检索文档并回答..."):
        res = qa_chain.invoke({"query": user_query})
    
    # 输出回答
    st.chat_message("user").write(user_query)
    st.chat_message("assistant").write(res["result"])

    # 展示引用原文片段
    with st.expander("📎 检索参考原文"):
        for idx, doc in enumerate(res["source_documents"]):
            st.markdown(f"**片段{idx+1}：**")
            st.write(doc.page_content)