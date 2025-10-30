import os
import json
import re
import operator
import copy
from typing import List, Optional, Dict, TypedDict, Annotated, Any
from pprint import pprint

# --- ИМПОРТЫ LANGCHAIN & LANGGRAPH ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from . import DocAgentTools as t

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
# ===============================================================
# Шаг 3: Собираем граф с новым узлом для сохранения
# ===============================================================

# --- 3.1: Обновляем состояние графа ---
class GraphState(TypedDict):

    file_paths: List[str]
    input_course_json: dict 
    
    all_source_text: str  
    
    populated_course: Optional[dict]
    output_file_path: Optional[str] 

def parse_files_node(state: GraphState) -> Dict[str, Any]:
    """
    Узел 1: Читает все файлы, выбирая парсер из словаря-диспетчера.
    """
    print("--- Узел 1: Парсинг файлов ---")

    # 1. Создаем словарь-диспетчер (карту расширений и функций)
    parsers = {
        ".pdf": t.parse_pdf,
        ".docx": t.parse_docx,
        ".txt": t.parse_txt
        # Добавляйте сюда новые парсеры по мере необходимости
    }
    
    all_texts = []
    for file_path in state['file_paths']:
        try:
            # 2. Получаем расширение файла
            _, extension = os.path.splitext(file_path)
            extension = extension.lower()
            
            # 3. Находим нужный парсер в словаре
            if extension in parsers:
                parser_function = parsers[extension]
                # 4. Вызываем найденную функцию
                text = parser_function(file_path)
                all_texts.append(text)
            else:
                print(f"  - ВНИМАНИЕ: Не найден парсер для файла {file_path} (расширение {extension})")
                all_texts.append(f"Не удалось обработать файл: {file_path}")

        except Exception as e:
            print(f"  - ОШИБКА при обработке файла {file_path}: {e}")
            all_texts.append(f"Ошибка при обработке файла: {file_path}")
    
    combined_text = "\n\n--- НОВЫЙ ИСТОЧНИК ---\n\n".join(all_texts)
    return {"all_source_text": combined_text}


def fill_content_node(state: GraphState) -> Dict[str, Any]:
    """
    Узел 2: Рекурсивно СОЗДАЕТ НОВЫЙ JSON, заполняя поля 'content' с сохранением структуры.
    """
    print("--- Узел 2: Заполнение контента курса ---")
    
    course_skeleton = state['input_course_json']
    all_source_text = state['all_source_text']

    prompt = ChatPromptTemplate.from_template(
        "Ты — автор образовательных материалов. Напиши краткое, но содержательное описание для раздела курса.\n\n"
        "Тема раздела: '{topic_title}'\n\n"
        "Используй ТОЛЬКО следующую информацию из источников. Если информации нет, напиши 'Информации по данной теме в источниках не найдено.'.\n\n"
        "--- ИСТОЧНИКИ ---\n{context}\n\n"

    )
    content_filler_chain = prompt | llm | StrOutputParser()

    # Новая рекурсивная функция, которая ВОЗВРАЩАЕТ результат
    def _build_and_fill_recursive(skeleton_topics: List[Dict]) -> List[Dict]:
        populated_topics = []
        for skeleton_topic in skeleton_topics:
            topic_title = skeleton_topic.get("title", "Без названия")
            print(f"  - Заполнение темы: {topic_title}")

            # 1. Генерируем контент для текущей темы
            generated_content = content_filler_chain.invoke({
                "topic_title": topic_title,
                "context": all_source_text
            }).strip()
            
            # 2. Рекурсивно обрабатываем вложенные темы
            populated_sub_topics = []
            if "sub_topics" in skeleton_topic and skeleton_topic["sub_topics"]:
                populated_sub_topics = _build_and_fill_recursive(skeleton_topic["sub_topics"])
            
            # 3. Собираем НОВЫЙ словарь с ЯВНЫМ ПОРЯДКОМ КЛЮЧЕЙ
            new_topic = {
                "title": topic_title,
                "content": generated_content,
                "sub_topics": populated_sub_topics
            }
            populated_topics.append(new_topic)
            
        return populated_topics

    # Запускаем рекурсивное построение
    populated_chapters = []
    if "chapters" in course_skeleton:
        populated_chapters = _build_and_fill_recursive(course_skeleton["chapters"])
        
    # Собираем финальный объект, сохраняя порядок ключей верхнего уровня
    final_populated_course = {
        "course_title": course_skeleton.get("course_title", "Без названия"),
        "chapters": populated_chapters
    }
        
    return {"populated_course": final_populated_course}


# --- НОВЫЙ УЗЕЛ: Сохранение результата в JSON-файл ---
def save_course_node(state: GraphState) -> Dict[str, str]:
    """
    Узел 3: Сохраняет итоговый заполненный JSON в файл.
    """
    print("--- Узел 3: Сохранение итогового файла ---")
    populated_course = state.get("populated_course")

    if not populated_course:
        print("Ошибка: курс не был заполнен, нечего сохранять.")
        return {}

    course_title = populated_course.get("course_title", "untitled_course").replace(' ', '_').lower()
    output_filename = f"{course_title}_final.json"
    
    course_to_save = {
        "course_title": populated_course.get("course_title"),
        "chapters": populated_course.get("chapters"),
        "output_file_path": output_filename  # Добавляем путь ПЕРЕД сохранением
    }
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(course_to_save, f, indent=2, ensure_ascii=False)

        
    print(f"Курс успешно сохранен в файл: {output_filename}")
    return {"output_file_path": output_filename}


# --- 3.3: Собираем и связываем граф ---
workflow = StateGraph(GraphState)

workflow.add_node("parse_files", parse_files_node)
workflow.add_node("fill_content", fill_content_node)
workflow.add_node("save_json", save_course_node)

workflow.set_entry_point("parse_files")
workflow.add_edge("parse_files", "fill_content")
workflow.add_edge("fill_content", "save_json")
workflow.add_edge("save_json", END)

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