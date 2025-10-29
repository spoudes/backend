import re


def validate_liascript(content: str) -> tuple[bool, str]:
    """
    Валидация LiaScript разметки
    
    Args:
        content: Строка с LiaScript разметкой
        
    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    errors = []
    
    # Проверка наличия заголовков разделов
    sections = re.findall(r'^#{1,6}\s+.+$', content, re.MULTILINE)
    if not sections:
        errors.append("No section headers found (lines starting with #)")
    
    # Проверка корректности single-choice quiz синтаксиса [( )] и [(X)]
    single_choice_options = re.findall(r'^\s*\[\([X\s]\)\]', content, re.MULTILINE)
    
    # Проверяем, что X в верхнем регистре
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
    quiz_lines = re.findall(r'^(\s*)\[\([X\s]\)\].*$', content, re.MULTILINE)
    quiz_lines += re.findall(r'^(\s*)\[\[[X\s]\]\].*$', content, re.MULTILINE)
    
    if quiz_lines:
        # Проверяем, что есть отступ
        no_indent = [line for line in quiz_lines if line == '']
        if no_indent:
            errors.append(
                f"Found {len(no_indent)} quiz options without indentation. Use 4 spaces before quiz options")
    
    # Проверка на неправильный синтаксис
    if '[Choices]' in content or '[/Choices]' in content:
        errors.append(
            "Found [Choices] tags - this is not valid LiaScript syntax. Use indented [( )] or [[ ]] instead")
    
    if '[quiz]' in content or '[/quiz]' in content:
        errors.append(
            "Found [quiz] tags - this is not valid LiaScript syntax. Questions should be plain text followed by indented options")
    
    # Проверка на вопросы в скобках (частая ошибка)
    questions_in_brackets = re.findall(r'^\[([^\]]+)\]\s*$', content, re.MULTILINE)
    invalid_questions = [q for q in questions_in_brackets
                         if not re.match(r'^\([\sX]\)$', q)
                         and not re.match(r'^\[[\sX]\]$', q)
                         and len(q) > 10]
    
    if invalid_questions:
        errors.append(
            f"Found {len(invalid_questions)} potential questions wrapped in brackets. Question text should NOT be in brackets")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, "Valid LiaScript"


def clean_liascript(content: str) -> str:
    """
    Очистка и нормализация LiaScript разметки
    
    Args:
        content: Строка с LiaScript разметкой
        
    Returns:
        str: Очищенная разметка
    """
    # Убрать неправильные теги
    content = re.sub(r'\[/?Choices\]', '', content, flags=re.IGNORECASE)
    content = re.sub(r'\[/?quiz\]', '', content, flags=re.IGNORECASE)
    
    # Убрать вопросы в скобках формата [Question text?]
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        if re.match(r'^\s*\[([^\]]{10,})\]\s*$', line):
            question_text = re.match(r'^\s*\[([^\]]+)\]\s*$', line).group(1)
            if not re.match(r'^\([\sX]\)', question_text) and not re.match(r'^\[[\sX]\]', question_text):
                cleaned_lines.append(question_text)
                continue
        
        cleaned_lines.append(line)
    
    content = '\n'.join(cleaned_lines)
    
    # Нормализовать отступы для опций викторин (4 пробела)
    lines = content.split('\n')
    normalized_lines = []
    
    for line in lines:
        if re.match(r'^\s*\[\([X\s]\)\]', line) or re.match(r'^\s*\[\[[X\s]\]\]', line):
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
    content = re.sub(r'(\?)\n(\s*\[\()', r'\1\n\n\2', content)
    
    content = content.strip()
    
    return content