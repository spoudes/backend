import json
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# ============ Определение схем данных ============

class MultipleChoiceQuestion(BaseModel):
    """Вопрос множественного выбора"""
    question: str = Field(description="Текст вопроса")
    options: List[str] = Field(description="Список из 4 вариантов ответа")
    correct_answers: List[int] = Field(description="Индексы правильных ответов (0-3)")
    explanation: str = Field(description="Объяснение правильного ответа")
    difficulty: Literal["легкий", "средний", "сложный"] = Field(description="Уровень сложности")

    @field_validator('options')
    @classmethod
    def validate_options(cls, v):
        """Проверка количества вариантов ответа"""
        if len(v) < 2:
            raise ValueError('Должно быть минимум 2 варианта ответа')
        # Дополняем до 4 вариантов если нужно
        while len(v) < 4:
            v.append(f"Вариант {len(v) + 1}")
        return v[:4]  # Ограничиваем максимум 4 вариантами

    @field_validator('correct_answers')
    @classmethod
    def validate_correct_answers(cls, v):
        """Проверка корректности индексов"""
        if not v:
            return [0]  # По умолчанию первый вариант
        return [idx for idx in v if 0 <= idx <= 3]


class TrueFalseQuestion(BaseModel):
    """Вопрос типа верно/неверно"""
    statement: str = Field(description="Утверждение для оценки")
    correct_answer: bool = Field(description="Правильный ответ: True или False")
    explanation: str = Field(description="Объяснение правильного ответа")
    difficulty: Literal["легкий", "средний", "сложный"] = Field(description="Уровень сложности")


class OpenEndedQuestion(BaseModel):
    """Открытый вопрос"""
    question: str = Field(description="Текст открытого вопроса")
    sample_answer: str = Field(description="Пример правильного ответа")
    key_points: List[str] = Field(description="Ключевые моменты, которые должны быть в ответе")
    difficulty: Literal["легкий", "средний", "сложный"] = Field(description="Уровень сложности")

    @field_validator('key_points')
    @classmethod
    def validate_key_points(cls, v):
        """Проверка ключевых моментов"""
        if not v:
            return ["Основная идея", "Детали и примеры"]
        if len(v) < 2:
            v.append("Дополнительный аспект")
        return v


class QuestionSet(BaseModel):
    """Набор вопросов для конкретной темы"""
    topic: str = Field(description="Название темы")
    multiple_choice: List[MultipleChoiceQuestion] = Field(default_factory=list,
                                                          description="Вопросы множественного выбора")
    true_false: List[TrueFalseQuestion] = Field(default_factory=list, description="Вопросы верно/неверно")
    open_ended: List[OpenEndedQuestion] = Field(default_factory=list, description="Открытые вопросы")


# ============ LLM-агент ============

class QuizGeneratorAgent:
    """Агент для генерации качественных тестовых вопросов"""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.7):
        """Инициализация агента"""
        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model,
            temperature=temperature
        )

        # Создаем структурированный вывод
        self.question_generator = self.llm.with_structured_output(
            QuestionSet,
            method="json_mode"  # Используем JSON mode для большей надежности
        )

    def _create_prompt(self, content: str, topic: str, questions_count: dict) -> ChatPromptTemplate:
        """Создание промпта для генерации вопросов"""
        template = ChatPromptTemplate.from_messages([
            ("system", """Ты опытный педагог и методист. Создавай качественные тестовые задания в формате JSON.

СТРОГИЕ ТРЕБОВАНИЯ К ФОРМАТУ:

1. Вопросы множественного выбора (multiple_choice):
   - question: текст вопроса
   - options: массив из РОВНО 4 вариантов ответа (строки)
   - correct_answers: массив индексов правильных ответов [0-3]
   - explanation: объяснение (строка)
   - difficulty: "легкий", "средний" или "сложный"

2. Вопросы верно/неверно (true_false):
   - statement: утверждение (строка)
   - correct_answer: true или false (boolean)
   - explanation: объяснение (строка)
   - difficulty: "легкий", "средний" или "сложный"

3. Открытые вопросы (open_ended):
   - question: текст вопроса (строка)
   - sample_answer: пример ответа (строка)
   - key_points: массив из МИНИМУМ 2 ключевых пунктов (строки)
   - difficulty: "легкий", "средний" или "сложный"

ВАЖНО:
- Все поля обязательны
- options должен содержать ровно 4 элемента
- key_points должен содержать минимум 2 элемента
- difficulty только: "легкий", "средний", "сложный"
- Вопросы должны быть разнообразными и проверять разные аспекты темы"""),
            ("user", """Тема: {topic}

Учебный материал:
{content}

Создай следующее количество вопросов каждого типа:
- Множественный выбор: {mc_count}
- Верно/неверно: {tf_count}
- Открытые вопросы: {oe_count}

Распределение по сложности для каждого типа:
- Легкие: {easy_count}
- Средние: {medium_count}
- Сложные: {hard_count}

Верни результат в виде JSON объекта со структурой QuestionSet.""")
        ])

        return template

    def generate_questions(
            self,
            content: str,
            topic: str,
            questions_per_type: int = 3,
            difficulty_distribution: Optional[dict] = None,
            max_retries: int = 3
    ) -> QuestionSet:
        """
        Генерация набора вопросов с повторными попытками при ошибках

        Args:
            content: Учебный контент
            topic: Название темы
            questions_per_type: Количество вопросов каждого типа
            difficulty_distribution: Распределение сложности
            max_retries: Максимальное количество попыток при ошибке

        Returns:
            QuestionSet: Набор вопросов
        """
        # Установка распределения по умолчанию
        if difficulty_distribution is None:
            easy = questions_per_type // 3 or 1
            medium = questions_per_type // 3 or 1
            hard = questions_per_type - easy - medium
            difficulty_distribution = {
                "легкий": easy,
                "средний": medium,
                "сложный": hard
            }

        # Создание промпта
        prompt = self._create_prompt(content, topic, difficulty_distribution)
        chain = prompt | self.question_generator

        # Попытки генерации с обработкой ошибок
        for attempt in range(max_retries):
            try:
                result = chain.invoke({
                    "topic": topic,
                    "content": content,
                    "mc_count": questions_per_type,
                    "tf_count": questions_per_type,
                    "oe_count": questions_per_type,
                    "easy_count": difficulty_distribution.get("легкий", 1),
                    "medium_count": difficulty_distribution.get("средний", 1),
                    "hard_count": difficulty_distribution.get("сложный", 1)
                })

                # Проверка, что хотя бы один тип вопросов сгенерирован
                if not (result.multiple_choice or result.true_false or result.open_ended):
                    if attempt < max_retries - 1:
                        print(f"Попытка {attempt + 1}: Пустой результат, повторяем...")
                        continue
                    else:
                        print(f"ВНИМАНИЕ: для темы '{topic}' не удалось сгенерировать все вопросы")

                return result

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Попытка {attempt + 1} не удалась для темы '{topic}': {str(e)}")
                    print("Повторяем попытку...")
                else:
                    print(f"ОШИБКА генерации для темы '{topic}' после {max_retries} попыток: {str(e)}")
                    # Возвращаем пустой набор вопросов
                    return QuestionSet(
                        topic=topic,
                        multiple_choice=[],
                        true_false=[],
                        open_ended=[]
                    )

        # На случай если цикл завершился без return
        return QuestionSet(topic=topic, multiple_choice=[], true_false=[], open_ended=[])

    def process_course(
            self,
            course_data: dict,
            questions_per_topic: int = 3,
            difficulty_distribution: Optional[dict] = None
    ) -> dict:
        """Обработка всего курса"""
        all_questions = {
            "course_title": course_data["course_title"],
            "chapters": []
        }

        total_chapters = len(course_data["chapters"])

        for idx, chapter in enumerate(course_data["chapters"], 1):
            print(f"\nОбрабатываем главу {idx}/{total_chapters}: '{chapter['title']}'")

            chapter_questions = {
                "title": chapter["title"],
                # "questions": self.generate_questions(
                #     content=chapter["content"],
                #     topic=chapter["title"],
                #     questions_per_type=questions_per_topic,
                #     difficulty_distribution=difficulty_distribution
                # ).model_dump(),
                "sub_topics": []
            }

            # Обработка подтем
            sub_topics = chapter.get("sub_topics", [])
            if sub_topics:
                print(f"  ├─ Найдено подтем: {len(sub_topics)}")

                for sub_idx, sub_topic in enumerate(sub_topics, 1):
                    print(f"  ├─ Подтема {sub_idx}/{len(sub_topics)}: '{sub_topic['title']}'")

                    sub_topic_questions = self.generate_questions(
                        content=sub_topic["content"],
                        topic=sub_topic["title"],
                        questions_per_type=questions_per_topic,
                        difficulty_distribution=difficulty_distribution
                    ).model_dump()

                    chapter_questions["sub_topics"].append({
                        "title": sub_topic["title"],
                        "questions": sub_topic_questions
                    })

            all_questions["chapters"].append(chapter_questions)
            print(f"  └─ Глава '{chapter['title']}' обработана!")

        return all_questions

    def save_questions(self, questions: dict, pre_output_file: str):
        """Сохранение вопросов в JSON файл"""

        output_file = f"{pre_output_file}_question.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(questions, f, ensure_ascii=False, indent=2)

        print(f"\nВопросы успешно сохранены в {output_file}!")
        return output_file