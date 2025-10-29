import re


def validate_liascript(content: str) -> tuple[bool, str]:
    """Валидация LiaScript разметки"""
    errors = []

    # Проверка наличия заголовков разделов
    sections = re.findall(r'^#{1,6}\s+.+$', content, re.MULTILINE)
    if not sections:
        errors.append("No section headers found (lines starting with #)")

    # Проверка корректности single-choice quiz синтаксиса [( )] и [(X)]
    # Ищем опции викторин
    single_choice_options = re.findall(r'^\s*\[\([X\s]\)\]', content, re.MULTILINE)

    # Проверяем, что есть правильные X в верхнем регистре
    if single_choice_options:
        incorrect_x = re.findall(r'^\s*\[\([x]\)\]', content, re.MULTILINE)
        if incorrect_x:
            errors.append(
                f"Found {len(incorrect_x)} single-choice options with lowercase 'x'. Use uppercase 'X': [(X)]")

    # Проверка корректности multiple-choice quiz синтаксиса [[ ]] и [[X]]
    multiple_choice_options = re.findall(r'^\s*\[\[[X\s]\]\]', content, re.MULTILINE)

    if multiple_choice_options:
        incorrect_x = re.findall(r'^\s*\[\[[x]\]\]', content, re.MULTILINE)
        if incorrect_x:
            errors.append(
                f"Found {len(incorrect_x)} multiple-choice options with lowercase 'x'. Use uppercase 'X': [[X]]")

    # Проверка правильности отступов для опций викторин
    # Опции должны иметь отступ (обычно 4 пробела)
    quiz_lines = re.findall(r'^(\s*)\[\([X\s]\)\].*$', content, re.MULTILINE)
    quiz_lines += re.findall(r'^(\s*)\[\[[X\s]\]\].*$', content, re.MULTILINE)

    if quiz_lines:
        # Проверяем, что есть хоть какой-то отступ
        no_indent = [line for line in quiz_lines if line == '']
        if no_indent:
            errors.append(f"Found {len(no_indent)} quiz options without indentation. Use 4 spaces before quiz options")

    # Проверка баланса markdown заголовков (опционально)
    # Убедимся, что есть контент после заголовков
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if re.match(r'^#{1,6}\s+.+$', line):
            # Проверяем, что после заголовка есть хоть какой-то контент
            has_content = False
            for j in range(i + 1, min(i + 10, len(lines))):
                if lines[j].strip() and not re.match(r'^#{1,6}\s+.+$', lines[j]):
                    has_content = True
                    break
            if not has_content and i < len(lines) - 1:
                errors.append(f"Section header '{line[:50]}...' has no content following it")

    # Проверка правильности текстовых вопросов [[answer]]
    text_answers = re.findall(r'^\s*\[\[([^\]]+)\]\]\s*$', content, re.MULTILINE)
    if text_answers:
        # Проверяем, что текстовые ответы имеют отступ
        unindented = re.findall(r'^\[\[([^\]]+)\]\]\s*$', content, re.MULTILINE)
        if unindented:
            errors.append(f"Found {len(unindented)} text input answers without indentation")

    # Проверка на неправильный старый синтаксис
    if '[Choices]' in content or '[/Choices]' in content:
        errors.append("Found [Choices] tags - this is not valid LiaScript syntax. Use indented [( )] or [[ ]] instead")

    if '[quiz]' in content or '[/quiz]' in content:
        errors.append(
            "Found [quiz] tags - this is not valid LiaScript syntax. Questions should be plain text followed by indented options")

    # Проверка на вопросы в скобках (частая ошибка)
    questions_in_brackets = re.findall(r'^\[([^\]]+)\]\s*$', content, re.MULTILINE)
    # Исключаем валидные конструкции (quiz options и text inputs)
    invalid_questions = [q for q in questions_in_brackets
                         if not re.match(r'^\([\sX]\)$', q)
                         and not re.match(r'^\[[\sX]\]$', q)
                         and len(q) > 10]  # Вопросы обычно длиннее 10 символов

    if invalid_questions:
        errors.append(
            f"Found {len(invalid_questions)} potential questions wrapped in brackets. Question text should NOT be in brackets")

    if errors:
        return False, "; ".join(errors)

    return True, "Valid LiaScript"


def clean_liascript(content: str) -> str:
    """Очистка и нормализация LiaScript разметки"""

    # Убрать неправильные теги [Choices], [/Choices], [quiz], [/quiz]
    content = re.sub(r'\[/?Choices\]', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\[/?quiz\]', '', content, flags=re.IGNORECASE)

    # Убрать вопросы в скобках формата [Question text?]
    # Но НЕ трогать опции викторин [( )], [(X)], [[ ]], [[X]]
    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        # Если строка выглядит как вопрос в скобках (длинная и не опция викторины)
        if re.match(r'^\s*\[([^\]]{10,})\]\s*$', line):
            # Извлекаем текст без скобок
            question_text = re.match(r'^\s*\[([^\]]+)\]\s*$', line).group(1)
            # Проверяем, что это не опция викторины
            if not re.match(r'^\([\sX]\)', question_text) and not re.match(r'^\[[\sX]\]', question_text):
                cleaned_lines.append(question_text)
                continue

        cleaned_lines.append(line)

    content = '\n'.join(cleaned_lines)

    # Нормализовать отступы для опций викторин (приводим к 4 пробелам)
    lines = content.split('\n')
    normalized_lines = []

    for line in lines:
        # Если это опция викторины с неправильным отступом
        if re.match(r'^\s*\[\([X\s]\)\]', line) or re.match(r'^\s*\[\[[X\s]\]\]', line):
            # Убираем все отступы и добавляем 4 пробела
            normalized_line = '    ' + line.lstrip()
            normalized_lines.append(normalized_line)
        else:
            normalized_lines.append(line)

    content = '\n'.join(normalized_lines)

    # Убрать лишние пустые строки (больше 2 подряд)
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Убрать пробелы в конце строк
    content = '\n'.join(line.rstrip() for line in content.split('\n'))

    # Убедиться, что после вопросов есть пустая строка перед опциями
    content = re.sub(
        r'(\?)\n(\s*\[\()',
        r'\1\n\n\2',
        content
    )

    # Нормализовать отступы
    content = content.strip()

    return content
