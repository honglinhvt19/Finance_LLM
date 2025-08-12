import os
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from utils import load_config, setup_env, save_output

setup_env()
config = load_config()

def load_rag_chain():
    """Load RAG chain: retriever + LLM."""
    # Load embedding and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name=config["embedding_model"],  # Đã cập nhật thành mô hình tiếng Việt
        model_kwargs={"device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"}
    )
    project_root = os.path.dirname(os.path.dirname(__file__))
    persist_dir = os.path.join(project_root, config["vector_store_dir"])
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": config["k_retrieval"]})
    
    # Load LLM (cập nhật với mô hình tiếng Việt; PhoGPT hỗ trợ instruct)
    tokenizer = AutoTokenizer.from_pretrained(config["llm_model"])
    model = AutoModelForCausalLM.from_pretrained(config["llm_model"], device_map="auto")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"]
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # Prompt template cập nhật cho tiếng Việt (hướng dẫn rõ ràng hơn)
    template = """
    Bạn là chuyên gia tài chính. Sử dụng ngữ cảnh được lấy từ báo cáo tài chính để trả lời câu hỏi hoặc tóm tắt.
    Nếu không biết, hãy nói "Tôi không biết". Giữ câu trả lời ngắn gọn, tối đa 3 câu.
    Đối với tóm tắt, tập trung vào các chỉ số tài chính chính như tài sản, nợ phải trả, rủi ro.
    Câu hỏi: {question}
    Ngữ cảnh: {context}
    Trả lời:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    # Build chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

if __name__ == "__main__":
    rag_chain = load_rag_chain()
    # Example queries bằng tiếng Việt
    queries = [
        "Tóm tắt tổng tài sản của Vietcombank năm 2023?",
        "Rủi ro kinh doanh của BIDV là gì?"
    ]
    for question in queries:
        answer = rag_chain.invoke(question)
        print(f"Câu hỏi: {question}\nTrả lời: {answer}\n")
        save_output(question, answer, summary=question.startswith("Tóm tắt"))