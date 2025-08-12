import os
import json
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from utils import load_config, setup_env

setup_env()
config = load_config()

def preprocess_data(pdf_path):

    project_root = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(project_root, config["processed_chunks_dir"])
    output_file = os.path.join(output_dir, f"{os.path.basename(pdf_path).replace('.pdf', '')}_chunks.json")

    if os.path.exists(output_file):
        print(f"{output_file} đã tồn tại, bỏ qua tách chunk cho {pdf_path}.")
        with open(output_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        return [Document(page_content=chunk["page_content"], metadata=chunk["metadata"]) for chunk in chunks_data]
    
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        if not text.strip():  
            print(f"{pdf_path}: Scan-based, chạy OCR...")
            images = convert_from_path(pdf_path)
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img, lang='vie', config='--psm 6') or ""
    except Exception as e:
        print(f"Lỗi khi đọc {pdf_path}: {e}")
        return []

    if not text.strip():
        raise ValueError(f"Không trích xuất được văn bản từ {pdf_path}.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    chunks = text_splitter.split_text(text)
    if not chunks:
        raise ValueError(f"Không tạo được chunks từ {pdf_path}.")
    documents = [Document(page_content=chunk, metadata={"source": pdf_path}) for chunk in chunks]
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(project_root, config["processed_chunks_dir"])
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{os.path.basename(pdf_path).replace('.pdf', '')}_chunks.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents], f, ensure_ascii=False, indent=4)
    return documents

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    raw_dir = os.path.join(project_root, config["data_raw_dir"])
    for pdf_file in os.listdir(raw_dir):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(raw_dir, pdf_file)
            try:
                chunks = preprocess_data(pdf_path)
                print(f"Đã xử lý {pdf_file}: {len(chunks)} chunks")
            except Exception as e:
                print(f"Lỗi khi xử lý {pdf_file}: {e}")