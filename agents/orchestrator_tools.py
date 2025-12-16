import json
import os

def merge_course_data(course_file, questions_file, output_file=None):
    """
    Объединяет данные курса и вопросов в единую структуру.
    
    Args:
        course_file: путь к файлу или словарь с данными курса
        questions_file: путь к файлу или словарь с вопросами
        output_file: путь для сохранения результата (опционально)
    
    Returns:
        Объединенный словарь
    """
    # Загрузка данных курса
    print("DEBUG МЕРДЖИМ КУРС")
    if isinstance(course_file, dict):
        course_data = course_file
    elif isinstance(course_file, (str, os.PathLike)):
        with open(course_file, 'r', encoding='utf-8') as f:
            course_data = json.load(f)
    else:
        raise TypeError("course_file должен быть путем к файлу или словарем")
    
    # Загрузка данных вопросов
    if isinstance(questions_file, dict):
        questions_data = questions_file
    elif isinstance(questions_file, (str, os.PathLike)):
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
    else:
        raise TypeError("questions_file должен быть путем к файлу или словарем")
    
    def merge_chapter(course_chapter, questions_chapter):
        """Рекурсивно объединяет главу курса с вопросами"""
        merged = {
            'title': course_chapter['title'],
            'content': course_chapter['content'],
            'diagram': course_chapter.get('diagram'),
            'questions': {
                'multiple_choice': [],
                'true_false': [],
                'open_ended': []
            },
            'sub_topics': []
        }
        
        # Добавляем вопросы, если они есть
        if questions_chapter and 'questions' in questions_chapter:
            q = questions_chapter['questions']
            merged['questions'] = {
                'multiple_choice': q.get('multiple_choice', []),
                'true_false': q.get('true_false', []),
                'open_ended': q.get('open_ended', [])
            }
        
        # Обрабатываем подтемы
        for course_sub in course_chapter.get('sub_topics', []):
            questions_sub = None
            
            if questions_chapter and 'sub_topics' in questions_chapter:
                for qs in questions_chapter['sub_topics']:
                    if qs['title'] == course_sub['title']:
                        questions_sub = qs
                        break
            
            merged['sub_topics'].append(merge_chapter(course_sub, questions_sub))
        
        return merged
    
    # Создаем объединенную структуру
    merged_data = {
        'course_title': course_data['course_title'],
        'chapters': []
    }
    
    # Объединяем главы
    for course_chapter in course_data['chapters']:
        questions_chapter = None
        
        for qc in questions_data['chapters']:
            if qc['title'] == course_chapter['title']:
                questions_chapter = qc
                break
        
        merged_data['chapters'].append(merge_chapter(course_chapter, questions_chapter))
    
    # Сохраняем результат, если указан путь
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        print(f"Результат сохранен в {output_file}")
    
    return merged_data