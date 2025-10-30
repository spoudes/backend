import docx
from pypdf import PdfReader
from langchain.tools import tool


# @tool
def parse_txt(file_path: str) -> str:
    """
    Читает текстовый файл с автоопределением кодировки (UTF-8 или Windows-1251).
    """
    # Пробуем UTF-8
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.strip():  # Проверяем, что файл не пустой
                print(f"✅ {file_path}: UTF-8, {len(content)} символов")
                return content
    except UnicodeDecodeError:
        pass

    # Fallback на Windows-1251 (cp1251)
    try:
        with open(file_path, 'r', encoding='cp1251') as f:
            content = f.read()
            print(f"✅ {file_path}: Windows-1251 (cp1251), {len(content)} символов")
            return content
    except UnicodeDecodeError:
        pass

    # Последний fallback: latin-1 (никогда не падает)
    with open(file_path, 'r', encoding='latin-1') as f:
        content = f.read()
        print(f"⚠️ {file_path}: latin-1 (возможно некорректно), {len(content)} символов")
        return content
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
