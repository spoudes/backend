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
            model="mistral-large-2411",
            temperature=0.1,
            base_url=os.getenv('MISTRAL_BASE_URL'),
            api_key=os.getenv('MISTRAL_API_KEY')
        )

        self.gen_prompt = PromptTemplate(
            template="""
            Ты - профессиональный дизайнер информации и эксперт по визуализации учебных материалов.
            Твоя задача - превратить сухой текст в яркую, понятную и красивую схему на языке Mermaid.js.

            ВХОДНОЙ ТЕКСТ:
            {content}

            ---
            ШАГ 1: ВЫБЕРИ ЛУЧШИЙ ТИП ДИАГРАММЫ
            Не используй всегда mindmap! Выбирай исходя из смысла:
            1. Если текст описывает ЭТАПЫ, ИСТОРИЮ или ВРЕМЕННУЮ ШКАЛУ -> Используй `timeline` или `flowchart LR`.
            2. Если текст СРАВНИВАЕТ понятия (А против Б) -> Используй `graph TD` с подграфами (subgraph) или `quadrantChart`.
            3. Если это ПРОЦЕСС с решениями (Если... То...) -> Используй `flowchart TD` с ромбами решений.
            4. Если это ИЕРАРХИЯ -> Используй `mindmap` (но только если структура глубокая).
            
            ШАГ 2: СОЗДАЙ КРАСИВЫЙ КОД MERMAID
            
            Требования к стилю (ОБЯЗАТЕЛЬНО):
            1. ИСПОЛЬЗУЙ ЦВЕТА: Добавляй `style NodeId fill:#color,stroke:#color` для важных блоков.
               - Пример цветов: #f9f, #bbf, #dfd, #ffd (пастельные тона).
            2. ИСПОЛЬЗУЙ ФОРМЫ: 
               - `([Начало/Конец])` - скругленные края
               - `[Процесс]` - прямоугольник
               - `{{{{Выбор?}}}}` - ромб (в mermaid это фигурные скобки)
               - `((Ключевое понятие))` - круг
            3. ИСПОЛЬЗУЙ ИКОНКИ (FontAwesome): Добавляй `fa:fa-user`, `fa:fa-book`, `fa:fa-cogs` внутрь текста узла.
               Пример: `id1("fa:fa-book Учебник")`
            
            Технические требования:
            - ID узлов ТОЛЬКО латиница (A, B, Node1).
            - Текст в узлах в кавычках.
            - Если текст не содержит структуры для схемы - верни просто слово "SKIP".

            Пример хорошего Flowchart:
            graph TD
              Start([fa:fa-play Начало]) --> Step1[fa:fa-cog Обработка]
              Step1 --> Decision{{{{fa:fa-question Готово?}}}}
              Decision -- Да --> End([fa:fa-check Конец])
              Decision -- Нет --> Step1
              style Start fill:#dfd
              style End fill:#f9f
            
            Верни ТОЛЬКО валидный код диаграммы (без ``````).
            """,
            input_variables=["content"]
        )

        self.fix_prompt = PromptTemplate(
            template="""
                    Ты - строгий линтер кода Mermaid.js. 
                    Твоя задача - исправить синтаксические ошибки, не ломая визуальный стиль.

                    Ошибочный код:
                    {code}

                    Исправь:
                    1. Пробелы в ID узлов.
                    2. Незакрытые скобки.
                    3. Неправильный синтаксис форм (например, замени < > на {{ }} для ромбов, если нужно).

                    Верни ТОЛЬКО исправленный код.
                    """,
            input_variables=["code"]
        )

        self.gen_chain = self.gen_prompt | self.llm | StrOutputParser()
        self.fix_chain = self.fix_prompt | self.llm | StrOutputParser()

    @staticmethod
    def _basic_syntax_check(code: str) -> bool:
        if not code or len(code) < 10 or "SKIP" in code:
            return False
        # Расширенный список диаграмм
        valid_starts = ["graph", "flowchart", "sequenceDiagram", "classDiagram", "stateDiagram", "mindmap", "timeline",
                        "quadrantChart", "pie"]
        if not any(code.strip().startswith(k) for k in valid_starts):
            return False
        if code.count("{") != code.count("}"): return False
        return True

    async def generate_diagram(self, content: str) -> Optional[str]:
        try:
            raw_code = await self.gen_chain.ainvoke({"content": content[:3000]})

            # Чистка от маркдауна
            raw_code = raw_code.replace("``````", "").strip()

            if "SKIP" in raw_code:
                return None

            # Если валидация не прошла, пробуем починить
            if not self._basic_syntax_check(raw_code):
                print(f"Diagram syntax issue. Fixing...")
                raw_code = await self.fix_chain.ainvoke({"code": raw_code})
                raw_code = raw_code.replace("``````", "").strip()

            return raw_code

        except Exception as e:
            print(f"Error generating diagram: {e}")
            return None
