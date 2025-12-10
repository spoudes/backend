import os
import json
from typing import List, Optional, Dict, TypedDict, Annotated, Any
from pprint import pprint
import asyncio

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from .rag import CourseContentAgent

load_dotenv()

# ===============================================================
# Шаг 3: Собираем граф с новым узлом для сохранения
# ===============================================================
rag_agent = CourseContentAgent(persist_dir="./chroma_db")

# --- 3.1: Обновляем состояние графа ---
class GraphState(TypedDict):

    file_paths: List[str]
    input_course_json: dict   
    populated_course: Optional[dict]
    output_file_path: Optional[str] 

# ===============================================================
# Узлы графа (Nodes)
# ===============================================================

async def ingest_node(state: GraphState) -> Dict[str, Any]:
    """
    Узел 1: Загрузка документов (Ingestion).
    Вызывает OCR (Mistral) -> Chunking -> Embedding -> Vector Store.
    """
    print("\n--- Узел 1: Загрузка и обработка документов (OCR + Embeddings) ---")
    files = state['file_paths']
    
    if not files:
        print("ВНИМАНИЕ: Список файлов пуст!")
        return {}

    # Вызываем метод агента из rag.py
    try:
        await rag_agent.ingest_documents(files)
        print(f"Успешно обработано файлов: {len(files)}")
    except Exception as e:
        print(f"ОШИБКА при загрузке документов: {e}")
        # Здесь можно решить: прерывать выполнение или пытаться продолжить
        # raise e 
    
    return {} # Состояние не меняем, так как данные ушли в ChromaDB (side effect)


async def generate_content_node(state: GraphState) -> Dict[str, Any]:
    """
    Узел 2: Генерация контента.
    Рекурсивно обходит JSON, делает RAG-поиск, проверку безопасности и генерацию.
    """
    print("\n--- Узел 2: Генерация контента курса (RAG + Security Check) ---")
    course_skeleton = state['input_course_json']
    
    # Запускаем "умное" заполнение через агента
    # fill_course_structure внутри себя вызывает SecurityAgent для проверки контекста
    filled_course = await rag_agent.fill_course_structure(
        course_skeleton, 
        max_concurrency=3 # Количество параллельных запросов к LLM
    )
    
    return {"populated_course": filled_course}


def save_node(state: GraphState) -> Dict[str, Any]:
    """
    Узел 3: Сохранение результата в JSON.
    """
    print("\n--- Узел 3: Сохранение результата ---")
    populated_course = state.get("populated_course")
    return populated_course
    # if not populated_course:
    #     print("Ошибка: Курс пуст, нечего сохранять.")
    #     return {}
        
    # course_title = populated_course.get("course_title", "untitled").replace(" ", "_").lower()
    # output_filename = f"{course_title}_final.json"
    
    # # Формируем структуру для сохранения
    # course_to_save = {
    #     "course_title": populated_course.get("course_title"),
    #     "chapters": populated_course.get("chapters"),
    #     "output_file_path": output_filename
    # }

    # try:
    #     with open(output_filename, "w", encoding="utf-8") as f:
    #         json.dump(course_to_save, f, indent=2, ensure_ascii=False)
    #     print(f"Файл успешно сохранен: {os.path.abspath(output_filename)}")
    # except Exception as e:
    #     print(f"Ошибка сохранения файла: {e}")
        
    # return {"output_file_path": output_filename}


# ===============================================================
# Сборка графа
# ===============================================================

workflow = StateGraph(GraphState)

workflow.add_node("ingest_docs", ingest_node)
workflow.add_node("generate_content", generate_content_node)
workflow.add_node("save_result", save_node)

workflow.set_entry_point("ingest_docs")
workflow.add_edge("ingest_docs", "generate_content")
workflow.add_edge("generate_content", "save_result")
workflow.add_edge("save_result", END)

DocAndCourseAgent = workflow.compile()

# ===============================================================
# Шаг 4: Запуск и проверка результата
# ===============================================================

# with open("doc1.txt", "w", encoding="utf-8") as f:
#     f.write("Эйнштейн создал теорию относительности.")

# with open("doc2.txt", "w", encoding="utf-8") as f:
#     f.write("Менделеев создал таблицу химических элементов")

# user_files = ["doc1.txt", "doc2.txt"]

# user_course_struct = {
#     "course_title": "Великие ученые",
#     "chapters": [
#     {
#         "title": "Физики",
#         "content": "",
#         "sub_topics": [
#         {
#             "title": "Эйнштейн",
#             "content": "",
#             "sub_topics": []
#         }
#         ]
#     },
#     {
#         "title": "Химики",
#         "content": "",
#         "sub_topics": [
#         {
#             "title": "Менделеев",
#             "content": "",
#             "sub_topics": []
#         }
#         ]
#     }
#     ]
# }

# initial_state = {
#     "file_paths": user_files,
#     "input_course_json": user_course_struct
# }

# final_state = DocAndCourseAgent.invoke(initial_state)

# print("\n\n--- ИТОГОВОЕ СОСТОЯНИЕ ГРАФА ---")
# pprint(final_state, sort_dicts=False)
# Допустим получаем также uuid = 1 => сохраняем в courses_src в директорию course_1
