import os
import re
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


class DiagramGeneratorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="mistral-large-latest",
            temperature=0.1,
            base_url=os.getenv('MISTRAL_BASE_URL'),
            api_key=os.getenv('MISTRAL_API_KEY')
        )

        # --- Промпт Генерации ---
        self.gen_prompt = PromptTemplate(
            template="""
            Ты - эксперт по визуализации данных и преподаватель. Твоя задача - создать диаграмму на языке Mermaid.js для учебного материала.

            ВХОДНОЙ ТЕКСТ:
            {content}

            ЗАДАЧА:
            1. Проанализируй текст. Если он описывает процесс, алгоритм, иерархию, сравнение или структуру - создай диаграмму.
            2. Если текст слишком абстрактный или не содержит структуры для визуализации - верни слово "SKIP".
            3. Используй типы диаграмм: 
               - `graph TD` или `graph LR` (для процессов и структур)
               - `sequenceDiagram` (для взаимодействий)
               - `mindmap` (для иерархий идей)

            ТРЕБОВАНИЯ К MERMAID:
            - Идентификаторы узлов (Node ID) должны быть ТОЛЬКО на латинице, без пробелов (напр. A, Node1, Process_Start).
            - Текст внутри узлов бери в кавычки или скобки: id1["Текст на русском"]
            - Не используй сложные стили, только структуру.

            Верни ТОЛЬКО код диаграммы (без ```
            """,
            input_variables=["content"]
        )

        # --- Промпт Валидации (LLM Judge) ---
        self.fix_prompt = PromptTemplate(
            template="""
            Ты - строгий линтер кода Mermaid.js. 
            Твоя задача - исправить синтаксические ошибки в коде.

            Ошибочный код:
            {code}

            Типичные ошибки для исправления:
            1. Пробелы в ID узлов (напр. A B --> C должно быть A_B --> C).
            2. Спецсимволы в ID узлов (используй безопасные ID, а текст пиши в метках ["Text"]).
            3. Незакрытые скобки.

            Верни ТОЛЬКО исправленный валидный код Mermaid.js.
            """,
            input_variables=["code"]
        )

        self.gen_chain = self.gen_prompt | self.llm | StrOutputParser()
        self.fix_chain = self.fix_prompt | self.llm | StrOutputParser()

    @staticmethod
    def _basic_syntax_check(code: str) -> bool:
        """
        Простая эвристика: проверяем баланс скобок и запрещенные символы в начале строк.
        """
        if not code or len(code) < 10:
            return False

        # Проверка ключевых слов
        valid_starts = ["graph", "flowchart", "sequenceDiagram", "classDiagram", "stateDiagram", "mindmap"]
        if not any(code.strip().startswith(k) for k in valid_starts):
            return False

        # Проверка баланса скобок (грубая)
        if code.count("{") != code.count("}"):
            return False
        if code.count("[") != code.count("]"):
            return False

        return True

    async def generate_diagram(self, content: str) -> Optional[str]:
        """
        Генерирует диаграмму, валидирует её и возвращает код.
        Возвращает None, если диаграмма не нужна или генерация не удалась.
        """
        # 1. Генерация
        try:
            # Берем первые 2000 символов, чтобы не перегружать контекст, если глава огромная
            raw_code = await self.gen_chain.ainvoke({"content": content[:2000]})
            raw_code = raw_code.strip()

            # Убираем markdown обертку, если LLM всё же её добавила
            raw_code = raw_code.replace("```mermaid", "").replace("```", "")

            if "SKIP" in raw_code or len(raw_code) < 10:
                return None

        except Exception as e:
            print(f"Error generating diagram: {e}")
            return None

        # 2. Валидация и исправление (Self-Correction)
        if not self._basic_syntax_check(raw_code):
            print(f"Diagram syntax check failed. Attempting auto-fix...")
            try:
                fixed_code = await self.fix_chain.ainvoke({"code": raw_code})
                fixed_code = fixed_code.replace("```mermaid", "").replace("```", "")
                return fixed_code
            except Exception as e:
                print(f"Error fixing diagram: {e}")
                return None  # Если даже фикс упал, лучше не возвращать битый код

        return raw_code
