import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

from agents.ContentCourseAgent.ContentAgentClass import CourseContentAgent
from agents.ContentCourseAgent.ContentAgentGraph import create_doc_agent_graph
from agents.DiagramAgent import DiagramGeneratorAgent
from agents.QuizAgent.quiz_generator import QuizGeneratorAgent
from agents.liascript_generator import generate_liascript_from_json, parse_json_to_liascript
from agents.liascript_generator.validate_and_cleanify import validate_liascript, clean_liascript
from agents.orchestrator_tools import merge_course_data
import copy
app = FastAPI(title="MultiAgent System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

quiz_agent_instance = QuizGeneratorAgent(
    api_key=os.getenv("MISTRAL_API_KEY"),
    model="gemini-2.5-flash",
    temperature=0.7
)

diagram_agent_instance = DiagramGeneratorAgent()


async def process_diagrams_recursive(node: dict[str, Any], agent: DiagramGeneratorAgent):
    """
    Рекурсивно обходит структуру курса и добавляет диаграммы.
    """
    tasks = []

    if 'content' in node and node['content']:
        async def _gen_for_node(n):
            diag_code = await agent.generate_diagram(n['content'])
            if diag_code:
                n['diagram'] = diag_code

        tasks.append(_gen_for_node(node))

    if 'chapters' in node:
        for chapter in node['chapters']:
            tasks.append(process_diagrams_recursive(chapter, agent))

    if 'sub_topics' in node:
        for sub in node['sub_topics']:
            tasks.append(process_diagrams_recursive(sub, agent))

    if tasks:
        await asyncio.gather(*tasks)

    return node

@app.post("/upload-files")
async def upload_files_for_course(request: Request):
    try:
        form = await request.form()

        course_structure_str = form.get("course_structure")
        if not course_structure_str:
            raise HTTPException(status_code=400, detail="Отсутствует course_structure")

        course_structure = json.loads(course_structure_str)

        uuid_for_course_folder = uuid4()
        course_dir = UPLOAD_DIR / str(uuid_for_course_folder)
        course_dir.mkdir(parents=True, exist_ok=True)
        with open(course_dir / "user_course.json", "w", encoding="utf-8") as f:
            json.dump(course_structure, f, ensure_ascii=False, indent=4)
        chapter_files: dict[int, list] = defaultdict(list)

        chapter_keys = set()
        for key in form.keys():
            if key.startswith("chapter_") and key.endswith("_files"):
                chapter_keys.add(key)

        for key in chapter_keys:
            try:
                chapter_index = int(key.split("_")[1])
                files = form.getlist(key)
                chapter_files[chapter_index].extend(files)
            except (IndexError, ValueError):
                continue

        chapters_info = []

        for idx, chapter_data in enumerate(course_structure.get("chapters", [])):
            chapter_info = {
                "index": idx,
                "title": chapter_data.get("title"),
                "files": []
            }
            if idx in chapter_files:
                chapter_subdir = course_dir / f"chapter_{idx}"
                chapter_subdir.mkdir(exist_ok=True)

                for upload_file in chapter_files[idx]:
                    file_path = chapter_subdir / upload_file.filename
                    with open(file_path, "wb") as f:
                        content = await upload_file.read()
                        f.write(content)

                    print(f"Сохранен файл: {upload_file.filename} ({len(content)} bytes)")

                    chapter_info["files"].append({
                        "filename": upload_file.filename,
                        "path": str(file_path),
                        "size": len(content),
                        "content_type": upload_file.content_type
                    })

            chapters_info.append(chapter_info)

        return {
            "status": "success",
            "message": "Курс успешно получен",
            "folder_id": uuid_for_course_folder,
            "chapters_count": len(course_structure.get("chapters", [])),
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка парсинга JSON: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.get("/generate-course/{folder_id}")
async def generate_course(folder_id: str):
    folder_path = Path(f'uploaded_files/{folder_id}')
    output_md_path = folder_path / 'course_final.md'

    if output_md_path.exists():
        print(f"Возвращаем кэшированный курс для {folder_id}")
        with open(output_md_path, 'r', encoding='utf-8') as f:
            return Response(
                content=f.read(),
                media_type="text/markdown; charset=utf-8"
            )

    user_files = [str(p) for p in folder_path.rglob('*') if p.is_file() and p.suffix != '.json' and p.suffix != '.md']
    json_path = folder_path / 'user_course.json'

    if not json_path.exists():
        return Response("Error: user_course.json not found", status_code=404)

    with open(json_path, 'r', encoding="utf-8") as fil:
        user_course_struct = json.load(fil)

    print("--- Запуск генерации контента (Agent 1) ---")
    content_agent = CourseContentAgent(persist_dir="./chroma_db")
    course_workflow = create_doc_agent_graph(content_agent)

    initial_state = {
        "file_paths": user_files,
        "input_course_json": user_course_struct,
        "populated_course": None,
        "output_file_path": None
    }

    workflow_result = await course_workflow.ainvoke(initial_state)
    populated_course = workflow_result['populated_course']

    if not populated_course:
        return Response("Error: Failed to generate course content", status_code=500)

    # 3. Этап 2: Параллельная генерация (Questions + Diagrams)
    print("--- Запуск параллельной генерации (Quiz + Diagrams) ---")
    print(f"Изначальный курс от course generator: {populated_course}")
    # Задача А: Викторина
    async def run_quiz_gen():
        difficulty_dist = {"легкий": 1, "средний": 1, "сложный": 1}
        questions = await quiz_agent_instance.process_course(
            course_data=populated_course,
            questions_per_topic=3,
            difficulty_distribution=difficulty_dist
        )
        return questions

    populated_course_for_diagrams = copy.deepcopy(populated_course)

    async def run_diagram_gen():
        await process_diagrams_recursive(populated_course_for_diagrams, diagram_agent_instance)
        return populated_course_for_diagrams

    results = await asyncio.gather(
        run_quiz_gen(),
        run_diagram_gen()
    )

    generated_questions = results[0]
    course_with_diagrams = results[1]
    print(f"generated questions: {generated_questions}")
    print(f"course with diagrams: {course_with_diagrams}")
    print("--- Сборка результатов ---")
    final_json_data = merge_course_data(
        course_file=course_with_diagrams,
        questions_file=generated_questions,
        output_file=None
    )

    print(f"json data to generate: {final_json_data}")
    liascript_markup = generate_liascript_from_json(json_data=final_json_data,
                                                    use_ai_validation=False)

    is_valid, message = validate_liascript(liascript_markup)

    if is_valid:
        print(f"✓ Валидация пройдена: {message}")
    else:
        liascript_markup = clean_liascript(liascript_markup)
        is_valid, message = validate_liascript(liascript_markup)
        print(f"После очистки: {message}")
    print("ПРОВАЛИДИРОВАЛ")
    await content_agent.shutdown()
    print("ЗАШАТДАУНИЛИ")
    return Response(
        content=liascript_markup,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Type": "text/markdown; charset=utf-8",
        }
    )


