import json
from collections import defaultdict
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request

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
        course_title = course_structure.get("course_title", "Untitled")
        uuid_for_course_folder = uuid4()
        course_dir = UPLOAD_DIR / uuid_for_course_folder
        course_dir.mkdir(parents=True, exist_ok=True)

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
            "chapters": chapters_info
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка парсинга JSON: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.post("/generate-course/{folder_id}")
def generate_course(request: Request, folder_id: str):
    # в этом роуте будет происходить вызов агентов
    # пока моковый ответ
    return {"folder_id": folder_id}