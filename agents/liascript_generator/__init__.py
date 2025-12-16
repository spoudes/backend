import json
import zlib
import base64
from langchain_core.runnables import Runnable
from .llm import simple_chain


def encode_kroki_url(mermaid_code: str) -> str:
    """Конвертирует код Mermaid в URL картинки через сервис Kroki.io"""
    if not mermaid_code or not mermaid_code.strip():
        return ""
    data = mermaid_code.strip().encode('utf-8')
    compressed = zlib.compress(data, 9)
    encoded = base64.urlsafe_b64encode(compressed).decode('utf-8')
    return f"https://kroki.io/mermaid/svg/{encoded}"


def parse_json_to_liascript(json_data):
    """
    Парсит JSON и генерирует LiaScript разметку

    Args:
        json_data: Строка JSON или словарь Python

    Returns:
        str: Сгенерированная LiaScript разметка
    """
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data

    liascript_content = []

    liascript_content.append(f"# {data['course_title']}\n\n")

    for chapter in data['chapters']:
        liascript_content.append(f"## {chapter['title']}\n\n")

        if chapter.get('content'):
            liascript_content.append("### Учебный материал\n\n")
            liascript_content.append(f"{chapter['content']}\n\n")

        diagram = chapter.get('diagram')
        if diagram and diagram.strip() != "SKIP":
            liascript_content.append("### Визуализация\n\n")
            img_url = encode_kroki_url(diagram.strip())
            liascript_content.append(f"![Схема]({img_url})\n\n")

        # Обрабатываем вопросы
        questions = chapter.get('questions', {})
        if any(questions.values()):
            liascript_content.append("### Вопросы для проверки\n\n")

            # Multiple choice вопросы
            for mc_q in questions.get('multiple_choice', []):
                liascript_content.append(f"{mc_q['question']}\n\n")

                # Определяем тип: single или multiple choice
                is_single_choice = len(mc_q['correct_answers']) == 1

                for idx, option in enumerate(mc_q['options']):
                    is_correct = idx in mc_q['correct_answers']

                    if is_single_choice:
                        # Single choice: [( )] или [(X)]
                        marker = "[(X)]" if is_correct else "[( )]"
                    else:
                        # Multiple choice: [[X]] или [[ ]]
                        marker = "[[X]]" if is_correct else "[[ ]]"

                    liascript_content.append(f"    {marker} {option}\n")

                # Добавляем пояснение если есть
                if mc_q.get('explanation'):
                    liascript_content.append("*******\n")
                    liascript_content.append(f"{mc_q['explanation']}\n")
                    liascript_content.append("*******\n")

                liascript_content.append("\n")

            # True/False вопросы
            for tf_q in questions.get('true_false', []):
                liascript_content.append(f"{tf_q['statement']}\n\n")

                correct = tf_q['correct_answer']

                liascript_content.append(f"    [(X)] Верно\n" if correct else f"    [( )] Верно\n")
                liascript_content.append(f"    [( )] Неверно\n" if correct else f"    [(X)] Неверно\n")

                # Добавляем пояснение если есть
                if tf_q.get('explanation'):
                    liascript_content.append("*******\n")
                    liascript_content.append(f"{tf_q['explanation']}\n")
                    liascript_content.append("*******\n")

                liascript_content.append("\n")

            # Open-ended вопросы
            for oe_q in questions.get('open_ended', []):
                liascript_content.append(f"{oe_q['question']}\n\n")

                # Используем sample_answer как ожидаемый ответ
                if oe_q.get('sample_answer'):
                    liascript_content.append(f"    [[{oe_q['sample_answer']}]]\n")

                # Добавляем key_points как пояснение
                if oe_q.get('key_points'):
                    liascript_content.append("*******\n")
                    liascript_content.append("**Ключевые моменты:**\n\n")
                    for point in oe_q['key_points']:
                        liascript_content.append(f"- {point}\n")
                    liascript_content.append("*******\n")

                liascript_content.append("\n")

        # Рекурсивно обрабатываем подтемы
        for sub_topic in chapter.get('sub_topics', []):
            process_subtopic(sub_topic, liascript_content, level=3)

    return ''.join(liascript_content)


def process_subtopic(topic, content_list, level=3):
    """
    Рекурсивно обрабатывает подтемы

    Args:
        topic: Словарь с данными подтемы
        content_list: Список для добавления строк разметки
        level: Уровень заголовка (количество #)
    """
    header_prefix = '#' * level

    content_list.append(f"{header_prefix} {topic['title']}\n\n")

    # Учебный материал
    if topic.get('content'):
        content_list.append(f"{topic['content']}\n\n")

    diagram = topic.get('diagram')
    if diagram and diagram.strip() != "SKIP":
        content_list.append("### Визуализация\n\n")
        # Генерируем ссылку
        img_url = encode_kroki_url(diagram.strip())
        # Вставляем как картинку
        content_list.append(f"![Схема]({img_url})\n\n")

    # Обрабатываем вопросы подтемы
    questions = topic.get('questions', {})
    if any(questions.values()):
        sub_header = '#' * (level + 1)
        content_list.append(f"{sub_header} Вопросы для проверки\n\n")

        # Multiple choice вопросы
        for mc_q in questions.get('multiple_choice', []):
            content_list.append(f"{mc_q['question']}\n\n")

            is_single_choice = len(mc_q['correct_answers']) == 1

            for idx, option in enumerate(mc_q['options']):
                is_correct = idx in mc_q['correct_answers']

                if is_single_choice:
                    marker = "[(X)]" if is_correct else "[( )]"
                else:
                    marker = "[[X]]" if is_correct else "[[ ]]"

                content_list.append(f"    {marker} {option}\n")

            if mc_q.get('explanation'):
                content_list.append("*******\n")
                content_list.append(f"{mc_q['explanation']}\n")
                content_list.append("*******\n")

            content_list.append("\n")

        # True/False вопросы
        for tf_q in questions.get('true_false', []):
            content_list.append(f"{tf_q['statement']}\n\n")

            correct = tf_q['correct_answer']

            content_list.append(f"    [(X)] Верно\n" if correct else f"    [( )] Верно\n")
            content_list.append(f"    [( )] Неверно\n" if correct else f"    [(X)] Неверно\n")

            if tf_q.get('explanation'):
                content_list.append("*******\n")
                content_list.append(f"{tf_q['explanation']}\n")
                content_list.append("*******\n")

            content_list.append("\n")

        # Open-ended вопросы
        for oe_q in questions.get('open_ended', []):
            content_list.append(f"{oe_q['question']}\n\n")

            if oe_q.get('sample_answer'):
                content_list.append(f"    [[{oe_q['sample_answer']}]]\n")

            if oe_q.get('key_points'):
                content_list.append("*******\n")
                content_list.append("**Ключевые моменты:**\n\n")
                for point in oe_q['key_points']:
                    content_list.append(f"- {point}\n")
                content_list.append("*******\n")

            content_list.append("\n")

    # Рекурсивно обрабатываем вложенные подтемы
    for sub_topic in topic.get('sub_topics', []):
        process_subtopic(sub_topic, content_list, level=level+1)


# Упрощенный SYSTEM_PROMPT для ИИ-агента (теперь агент только проверяет и форматирует)
SYSTEM_PROMPT = (
    "You are an expert in reviewing and validating LiaScript markup.\n\n"
    
    "# Your Task\n"
    "You receive already generated LiaScript markup from a JSON file.\n"
    "Your job is to:\n"
    "1. Verify the markup is syntactically correct\n"
    "2. Ensure proper formatting and spacing\n"
    "3. Check that quiz syntax is valid\n"
    "4. Return the validated markup without changes (unless fixing critical syntax errors)\n\n"
    
    "# LiaScript Quiz Syntax (for validation)\n\n"
    
    "## Single-Choice Questions:\n"
    "```\n"
    "Question text?\n\n"
    "    [( )] Incorrect option\n"
    "    [(X)] Correct option\n"
    "    [( )] Another incorrect\n"
    "```\n\n"
    
    "## Multiple-Choice Questions:\n"
    "```\n"
    "Question text?\n\n"
    "    [[ ]] Incorrect\n"
    "    [[X]] Correct\n"
    "    [[X]] Also correct\n"
    "```\n\n"
    
    "## Text Input Questions:\n"
    "```\n"
    "Question text?\n\n"
    "    [[Expected answer]]\n"
    "```\n\n"
    
    "# Validation Rules\n"
    "- Quiz options MUST have exactly 4 spaces indentation\n"
    "- Use uppercase X: [(X)] or [[X]], never [(x)] or [[x]]\n"
    "- Questions should NOT be wrapped in brackets\n"
    "- Blank line between question and options\n"
    "- Explanations wrapped in stars: ****...**** \n\n"
    
    "Output ONLY the validated LiaScript markup."
)


def build_agent() -> Runnable:
    """
    Создает цепочку агента для валидации LiaScript
    """
    return simple_chain(SYSTEM_PROMPT)


def generate_liascript_from_json(json_data, use_ai_validation=True):
    """
    Главная функция для генерации LiaScript из JSON
    
    Args:
        json_data: JSON данные курса (строка или словарь)
        use_ai_validation: Использовать ли ИИ для валидации (опционально)
        
    Returns:
        str: Готовая LiaScript разметка
    """
    # Генерируем разметку из JSON
    liascript_markup = parse_json_to_liascript(json_data)
    print("DEBUG ГЕНЕРИРУЕМ ЛИАСКРИПТ")
    # Опционально: используем ИИ для валидации
    if use_ai_validation:
        agent = build_agent()
        validated_markup = agent.invoke({"input": liascript_markup})
        return validated_markup
    print("СГЕНЕРИРОВАН КУРС")
    return liascript_markup