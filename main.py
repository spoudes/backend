import json
import os
from collections import defaultdict
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

from agents.ContentCourseAgent.ContentAgentGraph import DocAndCourseAgent
from agents.QuizAgent.quiz_generator import QuizGeneratorAgent
from agents.liascript_generator import generate_liascript_from_json
from agents.liascript_generator.validate_and_cleanify import validate_liascript, clean_liascript
from agents.orchestrator_tools import merge_course_data

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
    folder_path = f'uploaded_files/{folder_id}'
    user_files = [str(p) for p in Path(folder_path).rglob('*') if p.is_file()]
    with open(f'uploaded_files/{folder_id}/user_course.json', 'r', encoding="utf-8") as fil:
        user_course_struct = json.load(fil)
    print(user_files)
    print(user_course_struct)

    initial_state = {
        "file_paths": user_files,
        "input_course_json": user_course_struct,
        "populated_course": None,
        "output_file_path": None
    }

    populated_course = await DocAndCourseAgent.ainvoke(initial_state)

    # ======Второй агент=======

    QuizAgent = QuizGeneratorAgent(api_key=os.getenv("GOOGLE_API_KEY"),
                                   model="gemini-2.5-flash",
                                   temperature=0.7)

    difficulty_dist = {
        "легкий": 1,
        "средний": 1,
        "сложный": 1
    }
    all_questions = QuizAgent.process_course(
        course_data=populated_course,
        questions_per_topic=3,  # 3 вопроса каждого типа
        difficulty_distribution=difficulty_dist
    )

    pre_out_put_filename = populated_course["course_title"]

    output_file = QuizAgent.save_questions(questions=all_questions,
                                           pre_output_file=pre_out_put_filename)

    with open(f"{output_file}", 'r', encoding='utf-8') as f:  # распаковываем question
        unpacked_question_file = json.load(f)

    new_output_file = f"{pre_out_put_filename}_merge.json"  # json для 3 агента с liaskript

    merge_output_file = merge_course_data(course_file=populated_course,
                                          questions_file=unpacked_question_file,
                                          output_file=new_output_file)

    liascript_markup = generate_liascript_from_json(json_data=merge_output_file,
                                                    use_ai_validation=True)


    is_valid, message = validate_liascript(liascript_markup)

    if is_valid:
        print(f"✓ Валидация пройдена: {message}")
    else:
        liascript_markup = clean_liascript(liascript_markup)
        is_valid, message = validate_liascript(liascript_markup)
        print(f"После очистки: {message}")

    return Response(
        content=liascript_markup,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Type": "text/markdown; charset=utf-8",
            "Access-Control-Allow-Origin": "*",
        }
    )


