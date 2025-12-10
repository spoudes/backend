import os
import json
from typing import List, Dict, Any
import copy
import asyncio
from datetime import datetime
import gc

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.messages import HumanMessage
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

from dotenv import load_dotenv

from agents.ContentCourseAgent.ContentAgentOCR import MistralOCRLoader
from agents.ContentCourseAgent.ContentAgentSec import SecurityAgent

load_dotenv()

class CourseContentAgent:

    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        
        self.trace_logs = []

        print("Загрузка модели эмбеддингов (это может занять время при первом запуске)...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # LLM (для генерации текста)
        self.text_llm = ChatOpenAI(
            model="mistral-medium",
            temperature=0.2,
            # convert_system_message_to_human=True,
            base_url=os.getenv('MISTRAL_BASE_URL'),
            api_key=os.getenv('MISTRAL_API_KEY')
        )

        self.pic_llm = ChatOpenAI(
            model="pixtral-12b-2409",
            temperature=0,
            base_url=os.getenv('MISTRAL_BASE_URL'),
            api_key=os.getenv('MISTRAL_API_KEY')
        )
        
        self.sec_llm = ChatOpenAI(
            model="mistral-small-2503",
            temperature=0.1,
            base_url=os.getenv('MISTRAL_BASE_URL'),
            api_key=os.getenv('MISTRAL_API_KEY')
        )

        self.sec_agent = SecurityAgent(self.sec_llm)

        self.vector_store = None
        self.retriever = None
        
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            print(f"Обнаружена существующая база в {persist_dir}. Загружаю...")
            self.vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5,
                                                                           })
        else:
            print("База не найдена. Будет создана новая при вызове ingest_documents.")
            self.vector_store = None
            self.retriever = None

    @retry(
        reraise=True, 
        stop=stop_after_attempt(10), # Пытаться 10 раз
        wait=wait_exponential(multiplier=1, min=2, max=60), # Ждать 2с, 4с, 8с...
        # Повторять только если ошибка связана с Rate Limit (код 429)
        retry=retry_if_exception_type(Exception) 
    )

    async def shutdown(self):
        """Явно освобождает ресурсы перед завершением работы."""
        print("--- Shutdown: Очистка ресурсов агента ---")
        
        # 1. Сбрасываем ссылки на объекты Chroma/Retriever
        self.vector_store = None
        self.retriever = None
        
        # 2. Если бы вы хранили сессии aiohttp явно, их надо было бы закрыть тут.
        # LangChain обычно закрывает их сам при удалении объекта, но gc.collect помогает.
        
        # 3. Принудительная сборка мусора
        gc.collect()
        print("--- Shutdown: Ресурсы очищены ---")

    async def _safe_text_llm_call(self, prompt):
        return await self.text_llm.ainvoke(prompt)

            # В utils.py или внутри CourseContentAgent
    async def describe_images_with_llm(self, pic_llm, images_dict):
        captions = {}
        print(f"Генерация описаний для {len(images_dict)} изображений...")
        
        for img_id, data in images_dict.items():
            b64_data = data['base64']
            
            # Создаем мультимодальное сообщение
            message = HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": "Опиши подробно, что изображено на этой картинке. Если это схема, опиши её структуру текстом."
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                    }
                ]
            )
            
            try:
                # Вызов ainvoke, передаем СПИСОК сообщений
                response = await pic_llm.ainvoke([message])
                captions[img_id] = response.content
                print(f"Img {img_id}: Описание готово:")
            except Exception as e:
                print(f"Ошибка Vision API для {img_id}: {e}")
                captions[img_id] = "Описание недоступно (ошибка модели)."
                
        return captions

    
    async def ingest_documents(self, file_paths: List[str]):
        """
        Считывает файлы, разбивает на чанки и создает векторный индекс.
        """
        ocr_loader = MistralOCRLoader(api_key=os.getenv('MISTRAL_API_KEY'))
        documents = []
        print(f"--- Начало обработки {len(file_paths)} файлов ---")

        for file_path in file_paths:
            try:
                markdown_text, images_data = ocr_loader.process_file(file_path)

                # captions = await self.describe_images_with_llm(self.pic_llm, images_data)


                processed_text = markdown_text
                # for img_id, caption in captions.items():
                #     original_ref = f"![{img_id}]({img_id})"
                #     new_ref = f"\n\n[ИЗОБРАЖЕНИЕ: {img_id}]\nОписание: {caption}\nПуть: {images_data[img_id]['path']}\n\n"
                #     processed_text = processed_text.replace(original_ref, new_ref)
                #     processed_text = processed_text.replace(img_id, f"{img_id} ({caption})")
                
                doc = Document(
                    page_content=processed_text,
                    metadata={"source": file_path}
                )
                documents.append(doc)

                # ext = file_path.split('.')[-1].lower()
                # loader = None
                # if ext == 'pdf':
                #     loader = PyPDFLoader(file_path)
                # elif ext in ['docx', 'doc']:
                #     loader = Docx2txtLoader(file_path)
                # elif ext in ['pptx', 'ppt']:
                #     loader = UnstructuredPowerPointLoader(file_path)
                # # Можно добавить txt, md и т.д.

            except Exception as e:
                print(f"Ошибка при чтении {file_path}: {e}")

        if not documents:
            raise ValueError("Не удалось загрузить ни одного документа.")

        # 2. Разбиение на чанки (Chunking)
        # Размер чанка важен: 1000 символов достаточно для контекста, overlap 200 сохраняет смысл на стыках
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        print(f"Создано {len(splits)} текстовых чанков.")

        # 3. Создание векторного хранилища (Chroma)
        # Persist_directory не указан, храним в памяти для скорости выполнения скрипта
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            self.vector_store.add_documents(splits)
        
        # Создаем ретривер (инструмент поиска)
        # k=5 означает, что мы берем 5 самых релевантных кусков текста для каждого запроса
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        print("--- Векторная база готова и сохранена на диск ---")

    async def _generate_content_node(self, node_title: str, context_hierarchy: List[str],
                                has_children: bool, children_titles_list: List[str] = None) -> str:
        """
        Генерирует контент для конкретного узла, используя RAG.
        context_hierarchy: список заголовков от корня до текущего узла (чтобы понимать контекст).
        """
        
        # Формируем поисковый запрос, включающий контекст родителей
        # Пример: "Курс: Великие Ученые. Глава: Физики. Тема: Эйнштейн"
        hierarchy_str = " > ".join(context_hierarchy)
        search_query = f"{hierarchy_str} > {node_title}"
        
        # 1. RAG: Поиск релевантных кусков
        relevant_docs = self.retriever.invoke(search_query)
        
        retrieved_contexts = [doc.page_content for doc in relevant_docs]

        # Собираем текст из найденных кусков
        context_text = "\n\n".join(retrieved_contexts)
        
        if not context_text:
            return "К сожалению, в предоставленных документах нет информации по этой теме."
        
        is_safe = await self.sec_agent.check_context(context_text)
        
        if not is_safe:
            self.trace_logs.append({
                "question": search_query,
                "answer": "[BLOCKED BY SECURITY]",
                "contexts": retrieved_contexts
            })
            return "Извините, найденная информация содержит контент, нарушающий политику безопасности, и не может быть использована."

        topics = ""

        if has_children:
        # 2. Промпт для LLM
            topics = ", ".join(children_titles_list) if children_titles_list else ""
            prompt_template = """
        Ты - редактор учебного курса. Твоя задача - написать ВВЕДЕНИЕ к разделу "{title}".
        
        В этом разделе будут изучены темы: {topics}.
        
        Используй найденный контекст, чтобы кратко упомянуть КЛЮЧЕВЫЕ достижения, но НЕ раскрывать их суть детально. 
        Сделай "тизер": назови ЧТО мы изучим, но не объясняй КАК это работает.

        НЕ ДУБЛИРУЙ НАЗВАНИЕ РАЗДЕЛА.

        Контекст из документов:
        ---
        {context}
        ---
        
        Пример ХОРОШЕГО ответа:
        "В этом разделе мы познакомимся с Альбертом Эйнштейном и узнаем, как его специальная теория относительности изменила представление о времени. Также мы рассмотрим..."
        
        
        Пример ПЛОХОГО ответа (спойлер):
        "Эйнштейн родился в 1879 году. Теория относительности гласит, что E=mc^2..." (Это слишком подробно для введения).
        
        Твоя задача: Написать 1 абзац (до 4 предложений).
        """
        else:
            # Для конечного урока
            prompt_template = """

            Ты пишешь материал конкретного урока "{title}".

            НЕ ДУБЛИРУЙ НАЗВАНИЕ УРОКА.

            Используй ТОЛЬКО следующую информацию из предоставленных документов для написания текста:
            {context}
            Не лей воду. Если факт один — напиши одно-два предложения.
            Не пиши "К сожалению информации мало".


            """

            # В КОНТЕКСТЕ МОГУТ БЫТЬ ИЗОБРАЖЕНИЯ В ФОРМАТЕ:
            # === ИЗОБРАЖЕНИЕ: img-X ===
            # [Описание: ...]
            # [Путь: ...]

            # ИНСТРУКЦИЯ ПО ИЗОБРАЖЕНИЯМ:
            # 1. Если описание картинки помогает раскрыть тему - используй информацию из описания в своем тексте.
            # 2. Чтобы вставить картинку в ответ, используй ТОЛЬКО специальный тег: <img src="ПУТЬ_ИЗ_КОНТЕКСТА">.
            # 3. Не копируй технические блоки "=== ИЗОБРАЖЕНИЕ ===" в финальный текст.
            # 4. Под картинкой напиши краткую подпись (caption), основанную на описании.

            # Пример ответа с картинкой:
            # "Как видно на схеме строения атома (img_physics_0_img-5.jpeg), электроны вращаются..."


        # prompt = PromptTemplate(
        #     template=prompt_template,
        #     input_variables=["hierarchy", "title", "context", "topics"]
        # )

        if has_children:
            final_prompt = prompt_template.format(
                # hierarchy=hierarchy_str,
                title=node_title,
                context=context_text,
                topics=topics
            )
        else:
            final_prompt = prompt_template.format(
            # hierarchy=hierarchy_str,
            title=node_title,
            context=context_text,
            # topics=topics
            )

        response = await self._safe_text_llm_call(final_prompt)

                # --- ЛОГИРОВАНИЕ ДЛЯ RAGAS ---
        # Мы сохраняем данные в список, не вызывая оценку сейчас
        self.trace_logs.append({
            "question": search_query,
            "answer": response.content,
            "contexts": retrieved_contexts, # Ragas требует список строк (list[str])
            # "ground_truth": "" # Для faithfulness и relevancy эталон не нужен
        })
        # -----------------------------
        return response.content

    async def _process_recursive(self, node: Dict[str, Any], hierarchy: List[str], semaphore: asyncio.Semaphore):
        """
        Рекурсивно обходит структуру и заполняет поле content.
        """
        current_title = node.get("title", node.get("course_title", "Untitled"))
        current_hierarchy = hierarchy + [current_title]
        
        # 1. Сначала определяем, есть ли дети и как их зовут
        children_titles = []
        # Ищем списки внутри узла (например, "chapters", "sub_topics")
        for key, value in node.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "title" in item:
                        children_titles.append(item["title"])
        
        has_children = len(children_titles) > 0
        
        tasks = []

        async def fill_current_node_content():
            # Если есть поле content и оно пустое - заполняем
            if "content" in node and not node["content"].strip():
                async with semaphore:
                    print(f"Генерация контента для: {' > '.join(current_hierarchy)}")
                    node["content"] = await self._generate_content_node(current_title,
                                                                hierarchy,
                                                                has_children,
                                                                children_titles_list=children_titles)
                
            # Проверяем вложенные структуры (chapters, sub_topics и т.д.)
            # Проходим по всем ключам, значения которых являются списками

        tasks.append(fill_current_node_content())

        for key, value in node.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        task = self._process_recursive(item, current_hierarchy, semaphore)
                        tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks)

    async def fill_course_structure(self, course_struct: Dict[str, Any], max_concurrency=5) -> Dict[str, Any]:
        """
        Основной метод запуска.
        """
        if not self.vector_store:
            raise RuntimeError("Сначала загрузите документы через ingest_documents()")
            
        # Клонируем структуру, чтобы не менять оригинал, если нужно
        result_struct = copy.deepcopy(course_struct)
        
        semaphore = asyncio.Semaphore(max_concurrency)
        # Запускаем рекурсию
        await self._process_recursive(result_struct, [], semaphore)
        
        return result_struct

    def regenerate_chapter(self, chapter_title, user_feedback):
        """
        Переписывает только одну главу, не пересоздавая базу.
        """
        if not self.vector_store:
            raise ValueError("База потеряна! Неоткуда брать информацию.")
            
        # 1. Ищем информацию в УЖЕ готовой базе (мгновенно, 0 рублей)
        docs = self.retriever.invoke(chapter_title)
        context = "\n".join([d.page_content for d in docs])
        
        # 2. Просим LLM переписать с учетом фидбека
        prompt = f"""
        Контекст: {context}
        Задача: Перепиши главу "{chapter_title}".
        Комментарий пользователя: "{user_feedback}" (например, "сделай проще, для детей").
        """
        return self.text_llm.invoke(prompt).content

    def evaluate_performance(self, output_dir="logs/ContentCourseAgent", output_file=f"quality_report{datetime.now().strftime("%H_%M_%S")}.json"):
        """
        Запускает RAGAS оценку на всех накопленных логах.
        """
        if not self.trace_logs:
            print("Нет данных для оценки (trace_logs пуст).")
            return

        print(f"--- Запуск оценки качества (RAGAS) для {len(self.trace_logs)} записей ---")
        print("Это может занять время, так как LLM будет проверять каждый ответ...")

        # 1. Подготовка данных в формате HuggingFace Dataset
        data = {
            "question": [log["question"] for log in self.trace_logs],
            "answer": [log["answer"] for log in self.trace_logs],
            "contexts": [log["contexts"] for log in self.trace_logs],
            # Добавляем ground_truth как пустые строки, так как у нас их нет,
            # но Ragas иногда требует наличие колонки (зависит от версии)
            # "ground_truth": [""] * len(self.trace_logs) 
        }
        
        dataset = Dataset.from_dict(data)

        # 2. Запуск оценки
        # Используем метрики, не требующие эталонного ответа (Ground Truth)
        # Faithfulness: не выдумал ли факты?
        # Answer Relevancy: ответил ли на поставленный вопрос?
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=self.text_llm,        # Используем ту же LLM (Llama-3 через OpenRouter)
            embeddings=self.embeddings
        )

        # 3. Вывод и сохранение
        print("\n=== Результаты оценки ===")
        print(results)

        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, output_file)

        # Конвертируем в Pandas DataFrame для удобства
        df = results.to_pandas()
        
        # Сохраняем в JSON (ориентация records удобна для чтения)
        df.to_json(full_path, orient="records", indent=2, force_ascii=False)
        print(f"Подробный отчет сохранен в: {os.path.abspath(full_path)}")
        
        # Возвращаем средние значения
        return results


# if __name__ == "__main__":

#     user_course_struct = {
#       "course_title": "Грокаем машинное обучение",
#       "chapters": [
#         {
#           "title": "Что такое машинное обучение?",
#           "content": "",
#           "sub_topics": [
#               { "title" : "Машинное обучение повсюду", "content": "" },
#               { "title" : "Так что же такое машинное обучение?", "content": "" },
#               { "title" : "Как заставить машины принимать решения с помощью данных?", "content": "" }
#           ]
#         },
#         {
#           "title": "Типы машинного обучения",
#           "content": "",
#           "sub_topics": [
#             { "title": "В чем разница между размеченными и неразмеченными данными", "content": "" },
#             { "title": "Контролируемое обучение", "content": "" },
#             { "title": "Неконтролируемое обучение", "content": "" },
#           ]
#         }
#       ]
#     }

    # user_course_struct = {
    #   "course_title": "Великие Ученые",
    #   "chapters": [
    #     {
    #       "title": "Физики",
    #       "content": "",
    #       "sub_topics": [
    #         { "title": "Эйнштейн", "content": "" }
    #       ]
    #     },
    #     {
    #       "title": "Химики",
    #       "content": "",
    #       "sub_topics": [
    #         { "title": "Менделеев", "content": "" }
    #       ]
    #     }
    #   ]
    # }

    

    # 3. Загрузка документов (замените пути на реальные файлы)
    # Для теста создайте simple.txt или используйте существующие pdf
    # agent.ingest_documents(["./books/physics_history.pdf", "./books/chemistry_basics.docx"])
    
    # !!! ДЛЯ ДЕМОНСТРАЦИИ (так как у меня нет ваших файлов) я создам фиктивный документ в памяти
    # В реальности вы просто вызовете ingest_documents с путями к файлам
    # class MockDocument:
    #     def __init__(self, content): self.page_content = content; self.metadata = {}
    
    # mock_text_physics = "Альберт Эйнштейн (1879—1955) — физик-теоретик, один из основателей современной теоретической физики. Разработал специальную теорию относительности."
    # mock_text_chem = "Дмитрий Иванович Менделеев — русский учёный-энциклопедист. Среди наиболее известных открытий — периодический закон химических элементов."
    
    # # Ручное создание базы для демо (в проде используйте ingest_documents)
    # agent.vector_store = Chroma.from_documents(
    #     [MockDocument(mock_text_physics), MockDocument(mock_text_chem)], 
    #     agent.embeddings
    # )
    # agent.retriever = agent.vector_store.as_retriever()
    # agent = CourseContentAgent()

    # asyncio.run(agent.ingest_documents(["./ML2060.pdf"]))
    # # 4. Заполнение структуры
    # filled_course = asyncio.run(agent.fill_course_structure(user_course_struct, max_concurrency=3))

    # # 5. Вывод результата
    # output_filename_title = str(filled_course.get("course_title")).strip()
    # output_filename = f"{output_filename_title}_course.json"

    # course_to_save = {
    #     "course_title": filled_course.get("course_title"),
    #     "chapters": filled_course.get("chapters"),
    #     "output_file_path": output_filename
    # }

    # with open(output_filename, "w", encoding="utf-8") as f:
    #     json.dump(course_to_save, f, indent=2, ensure_ascii=False)

    # # print(json.dumps(filled_course, indent=2, ensure_ascii=False))

    # agent.evaluate_performance("rag_quality_report.json")
