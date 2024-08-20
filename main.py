import streamlit as st
from utils import qa_agent
from langchain.memory import ConversationBufferMemory

st.header("📃 智能PDF问答工具")

with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥：", type="password")
    st.markdown("[获取OpenAI API密钥](https://api.aigc369.com/register?aff=87kh)")

#使用会话状态避免每次提交初始化memory
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(return_messages=True,
                                                          memory_key="chat_history",
                                                          output_key="answer")

uploaded_file = st.file_uploader("上传你的PDF文件：", type="pdf")
question = st.text_input("对PDF的内容进行提问", disabled=not uploaded_file)    #disabled控制输入框是否允许输入

if uploaded_file and question:
    if not openai_api_key:
        st.info("请输入你的OpenAI API密钥")
        st.stop()

    with st.spinner("AI正在思考中，请稍等..."):
        result = qa_agent(uploaded_file, question, st.session_state["memory"], openai_api_key)

    st.write("### 答案")
    st.write(result["answer"])
    #把历史对话放进记忆中
    st.session_state["chat_history"] = result["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"])-2:
                st.divider()