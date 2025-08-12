import streamlit as st
from rag_chain import load_rag_chain
from utils import save_output

rag_chain = load_rag_chain()

st.title("Tóm Tắt & QA Tài Chính")

question = st.text_input("Nhập câu hỏi hoặc yêu cầu tóm tắt tài chính (bằng tiếng Việt):")
if question:
    with st.spinner("Đang tạo câu trả lời..."):
        answer = rag_chain.invoke(question)
    st.write("**Trả lời:**")
    st.write(answer)
    save_output(question, answer, summary=question.lower().startswith("tóm tắt"))