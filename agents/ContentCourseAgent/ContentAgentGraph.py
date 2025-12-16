import os
import json
from typing import List, Optional, Dict, TypedDict, Annotated, Any
from pprint import pprint
import asyncio
import time

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv


load_dotenv()

class GraphState(TypedDict):

    file_paths: List[str]
    input_course_json: dict   
    populated_course: Optional[dict]
    output_file_path: Optional[str] 
    
# Узлы графа (Nodes)
def create_doc_agent_graph(rag_agent):

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

        try:
            await rag_agent.ingest_documents(files)
            print(f"Успешно обработано файлов: {len(files)}")
        except Exception as e:
            print(f"ОШИБКА при загрузке документов: {e}")
        
        return {}


    async def generate_content_node(state: GraphState) -> Dict[str, Any]:
        """
        Узел 2: Генерация контента.
        Рекурсивно обходит JSON, делает RAG-поиск, проверку безопасности и генерацию.
        """
        print("\n--- Узел 2: Генерация контента курса (RAG + Security Check) ---")
        course_skeleton = state['input_course_json']
    
        filled_course = await rag_agent.fill_course_structure(
            course_skeleton, 
            max_concurrency=3 # Количество параллельных запросов к LLM
        )
        
        return {"populated_course": filled_course}

    def evaluate_node(state: GraphState) -> str:
        """
        Узел 3: Проверка работы RAG с сохранением в ./logs/ContentCourseAgent.
        """
        print("\n--- Узел 3: Вычисление метрик ---")
        rag_agent.evaluate_performance()
        return {}
        


    def save_node(state: GraphState) -> Dict[str, Any]:
        """
        Узел 4: Сохранение результата в JSON.
        """
        print("\n--- Узел 4: Сохранение результата ---")
        populated_course = state.get("populated_course")
        return populated_course

    # Сборка графа

    workflow = StateGraph(GraphState)

    workflow.add_node("ingest_docs", ingest_node)
    workflow.add_node("generate_content", generate_content_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("save_result", save_node)

    workflow.set_entry_point("ingest_docs")
    workflow.add_edge("ingest_docs", "generate_content")
    workflow.add_edge("generate_content", "evaluate")
    workflow.add_edge("evaluate", "save_result")
    workflow.add_edge("save_result", END)

    return workflow.compile()
