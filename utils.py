from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

def qa_agent(uploaded_file, question, memory, api_key):
    #加载：把读取到内存的文件内容写入本地文件，然后把路径给加载器
    file_content = uploaded_file.read()   #返回bytes，二进制数据
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(file_content)

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    #切块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n", "。", "！", "？", "，", "、", ""]
    )
    texts = text_splitter.split_documents(docs)

    #嵌入
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large",
                                        openai_api_key=api_key,
                                        openai_api_base="https://api.aigc369.com/v1")

    #储存
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()

    #大模型
    model = ChatOpenAI(model="gpt-3.5-turbo",
                       openai_api_key=api_key,
                       openai_api_base="https://api.aigc369.com/v1")

    #对话链
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )

    result = qa.invoke({"chat_history": memory, "question": question})
    return result