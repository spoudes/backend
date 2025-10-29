import docx
from pypdf import PdfReader
from langchain.tools import tool


# @tool
def parse_txt(file_path: str) -> str:
    """
    Читает и возвращает текстовое содержимое из файла .txt.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# @tool
def parse_docx(file_path: str) -> str:
    """
    Читает и возвращает текстовое содержимое из файла .docx.
    """
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Ошибка при чтении DOCX: {e}"

def parse_pdf(pdf_path):
    """
    Извлекает текст из всех страниц PDF файла
    """
    reader = PdfReader(pdf_path)
    
    # Получаем количество страниц
    num_pages = len(reader.pages)
    print(f"Документ содержит {num_pages} страниц")
    
    # Извлекаем текст со всех страниц
    full_text = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        full_text.append(f"--- Страница {page_num + 1} ---\n{text}")
    
    return "\n\n".join(full_text)


# TOOLS = [
#     read_txt_file,
#     read_docx_file
# ]
