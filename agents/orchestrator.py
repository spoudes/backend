import json
import os
from dotenv import load_dotenv
from pprint import pprint

from DocAndCourseAgent.DocAgentMain import DocAndCourseAgent
from course_agent.quiz_generator import QuizGeneratorAgent
from orchestrator_tools import merge_course_data
from liascript_generator import generate_liascript_from_json
from liascript_generator.validate_and_cleanify import validate_liascript, clean_liascript


load_dotenv()
#================Запуск цепочки агентов==========
with open("doc1.txt", "w", encoding="utf-8") as f:
    f.write("Эйнштейн создал теорию относительности.")

with open("doc2.txt", "w", encoding="utf-8") as f:
    f.write("Менделеев создал таблицу химических элементов")

user_files = ["doc1.txt", "doc2.txt"]   #файлы с фронта

user_course_struct = {                  #Структура курса с фронта
    "course_title": "Великие ученые",
    "chapters": [
    {
        "title": "Физики",
        "content": "",
        "sub_topics": [
        {
            "title": "Эйнштейн",
            "content": "",
            "sub_topics": []
        }
        ]
    },
    {
        "title": "Химики",
        "content": "",
        "sub_topics": [
        {
            "title": "Менделеев",
            "content": "",
            "sub_topics": []
        }
        ]
    }
    ]
}

initial_state = {
    "file_paths": user_files,
    "input_course_json": user_course_struct
}

final_state = DocAndCourseAgent.invoke(initial_state) 

populated_course = final_state["populated_course"]

#======Второй агент=======

QuizAgent = QuizGeneratorAgent(api_key=os.getenv("GOOGLE_API_KEY"),
                               model="gemini-2.5-flash",
                               temperature=0.7)

difficulty_dist = {
    "легкий": 1,
    "средний": 1,
    "сложный": 1
}

print("=" * 60)
print("🚀 Начинаем генерацию вопросов...")
print("=" * 60)

all_questions = QuizAgent.process_course(
    course_data=populated_course,
    questions_per_topic=3,  # 3 вопроса каждого типа
    difficulty_distribution=difficulty_dist
)

pre_out_put_filename = populated_course["course_title"]

output_file = QuizAgent.save_questions(questions=all_questions,
                         pre_output_file=pre_out_put_filename)

with open(f"{output_file}", 'r', encoding='utf-8') as f: # распаковываем question
    unpacked_question_file = json.load(f)

new_output_file = f"{pre_out_put_filename}_merge.json" # json для 3 агента с liaskript

merge_output_file = merge_course_data(course_file=populated_course,
                                      questions_file=unpacked_question_file,
                                      output_file=new_output_file)


print("Генерация LiaScript разметки...")
liascript_markup = generate_liascript_from_json(json_data=merge_output_file,
                                                use_ai_validation=True)

print("Валидация разметки...")
is_valid, message = validate_liascript(liascript_markup)

if is_valid:
    print(f"✓ Валидация пройдена: {message}")
else:
    print(f"⚠ Обнаружены ошибки: {message}")
    print("Применяю автоматическую очистку...")
    liascript_markup = clean_liascript(liascript_markup)
    
    # Повторная валидация
    is_valid, message = validate_liascript(liascript_markup)
    print(f"После очистки: {message}")

output_file = 'course_output.md'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(liascript_markup)

print(f"\n✓ LiaScript разметка успешно сохранена в {output_file}")
print(f"Размер файла: {len(liascript_markup)} символов")

# Предварительный просмотр
print("\n" + "="*60)
print("ПРЕДВАРИТЕЛЬНЫЙ ПРОСМОТР:")
print("="*60)
print(liascript_markup[:500] + "...\n")



# with open("courses_src/course_1/великие_ученые_final.json", 'r', encoding='utf-8') as f:
#     data = json.load(f)


# pprint(data, sort_dicts=False)
