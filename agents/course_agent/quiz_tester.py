import json
import asyncio
import re
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============ Определение схем данных ============

class MetricScore(BaseModel):
    """Результат оценки по одной метрике"""
    name: str = Field(description="Название метрики")
    score: float = Field(description="Оценка (0-1)")
    reason: str = Field(description="Обоснование оценки")


class QuestionEvaluation(BaseModel):
    """Результат оценки одного вопроса"""
    question_type: str = Field(description="Тип вопроса: multiple_choice, true_false, open_ended")
    question_id: int = Field(description="Индекс вопроса")

    # GEval метрики
    relevance_metric: MetricScore = Field(description="Метрика релевантности")
    coherence_metric: MetricScore = Field(description="Метрика связности")
    conciseness_metric: MetricScore = Field(description="Метрика лаконичности")
    faithfulness_metric: MetricScore = Field(description="Метрика верности")

    # Детали
    issues: List[str] = Field(default_factory=list, description="Найденные проблемы")
    overall_score: float = Field(description="Общий скор (0-100)")
    recommendations: List[str] = Field(default_factory=list, description="Рекомендации")


class TopicEvaluation(BaseModel):
    """Результаты оценки всех вопросов по теме"""
    topic: str = Field(description="Название темы")
    total_questions: int = Field(description="Всего вопросов")

    multiple_choice_count: int
    true_false_count: int
    open_ended_count: int

    evaluations: List[QuestionEvaluation] = Field(description="Оценки вопросов")

    avg_relevance: float
    avg_coherence: float
    avg_conciseness: float
    avg_faithfulness: float
    overall_avg_score: float


class CourseEvaluation(BaseModel):
    """Итоговые результаты оценки курса"""
    course_title: str
    total_topics: int
    total_questions: int

    topic_evaluations: List[TopicEvaluation]

    overall_score: float
    quality_rating: str

    distribution: Dict[str, float]
    critical_issues: List[str]


# ============ GEval Metrics (Like DeepEval но для Gemini) ============

class GEvalMetric:
    """Базовый класс для GEval метрики"""

    def __init__(self, llm: ChatGoogleGenerativeAI, name: str, criteria: str):
        self.llm = llm
        self.name = name
        self.criteria = criteria

    async def evaluate(
        self,
        question: str,
        context: str,
        answer: str,
        semaphore: asyncio.Semaphore = None
    ) -> MetricScore:
        """Оценить вопрос по метрике"""
        async def _evaluate_with_semaphore():
            async with semaphore or asyncio.Semaphore(999):  # Если нет семафора - без ограничений
                prompt = ChatPromptTemplate.from_template("""Ты профессиональный оценщик качества образовательных материалов.
        
        Оцени следующий элемент по метрике "{name}" от 0 до 1.
        
        КРИТЕРИИ:
        {criteria}
        
        КОНТЕКСТ КУРСА:
        {context}
        
        ВОПРОС:
        {question}
        
        ОТВЕТ/ОБЪЯСНЕНИЕ:
        {answer}
        
        Ответь в формате JSON:
        {{
          "score": <число от 0 до 1>,
          "reason": "<краткое объяснение>"
        }}
        
        Ответ ТОЛЬКО JSON без markdown:""")

                chain = prompt | self.llm | StrOutputParser()

                try:
                    async with semaphore:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None,
                            lambda: chain.invoke({
                                "name": self.name,
                                "criteria": self.criteria,
                                "context": context[:400],
                                "question": question,
                                "answer": answer[:300]
                            })
                        )

                    # Парсим JSON
                    json_match = re.search(r'\{.*}', result, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group())
                        return MetricScore(
                            name=self.name,
                            score=float(data.get("score", 0.5)),
                            reason=data.get("reason", "")
                        )
                except Exception as e:
                    print(f"! Ошибка при оценке {self.name}: {e}")

                return MetricScore(
                    name=self.name,
                    score=0.5,
                    reason="Ошибка при оценке"
                )

        try:
            return await _evaluate_with_semaphore()
        except Exception as e:
            print(f"! Ошибка при оценке {self.name}: {e}")
            return MetricScore(name=self.name, score=0.5, reason="Ошибка при оценке")


# ============ Quiz Tester Agent с GEval ============

class QuizTesterAgentGEval:
    """Агент для оценки качества вопросов с использованием GEval метрик"""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", max_recurrent_requests: int = 3):
        """Инициализация тестера"""
        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model,
            temperature=0.0
        )

        self.semaphore = asyncio.Semaphore(max_recurrent_requests)

        # Инициализируем GEval метрики
        self.relevance_metric = GEvalMetric(
            self.llm,
            "Relevance",
            """Насколько вопрос релевантен и соответствует учебному контенту?
- 1.0: Идеально соответствует контенту, охватывает ключевые концепции
- 0.7: Хорошо соответствует, но может быть немного в стороне
- 0.5: Частично соответствует, есть связь с контентом
- 0.3: Слабо связан с контентом
- 0.0: Не соответствует контенту"""
        )

        self.coherence_metric = GEvalMetric(
            self.llm,
            "Coherence",
            """Насколько ясно и логично сформулирован вопрос?
- 1.0: Кристально ясно, логичная структура, легко понять
- 0.7: Ясно сформулировано, минимальные неясности
- 0.5: Понятно, но могут быть неоднозначности
- 0.3: Запутанно, требует переформулировки
- 0.0: Невозможно понять смысл"""
        )

        self.conciseness_metric = GEvalMetric(
            self.llm,
            "Conciseness",
            """Оптимальна ли длина и лаконичность вопроса?
- 1.0: Идеальная длина, никаких лишних слов, 0.5
- 0.7: Хорошая длина, может быть немного сокращено
- 0.5: Нормальная длина, приемлемо
- 0.3: Либо слишком длинно, либо слишком коротко
- 0.0: Экстремально длинный или короткий"""
        )

        self.faithfulness_metric = GEvalMetric(
            self.llm,
            "Faithfulness",
            """Насколько правильный ответ верен и соответствует контенту?
- 1.0: Полностью корректный, не содержит ошибок
- 0.7: Верно, но может быть неполно или упрощено
- 0.5: В целом верно, но есть неточности
- 0.3: Содержит ошибки или неточности
- 0.0: Кардинально неправильно"""
        )

    def _check_structural_issues(self, question: dict, q_type: str) -> List[str]:
        """Проверка структурных проблем вопроса"""
        issues = []

        if q_type == "multiple_choice":
            options = question.get("options", [])
            if len(options) != 4:
                issues.append(f"! Неправильное количество вариантов: {len(options)} вместо 4")

            if len(set(options)) != len(options):
                issues.append("! Есть дублирующиеся варианты ответов")

            correct = question.get("correct_answers", [])
            if not all(0 <= i <= 3 for i in correct):
                issues.append(f"! Неверные индексы правильных ответов: {correct}")

            q_len = len(question.get("question", "").split())
            if q_len < 5:
                issues.append(f"! Вопрос слишком короткий ({q_len} слов)")
            if q_len > 100:
                issues.append(f"! Вопрос слишком длинный ({q_len} слов)")

        elif q_type == "true_false":
            statement = question.get("statement", "")
            if statement.startswith("Неправда, что") or statement.count(" не ") > 2:
                issues.append("! Слишком много отрицаний в утверждении")

            if len(statement.split()) < 5:
                issues.append(f"! Утверждение слишком короткое")

        elif q_type == "open_ended":
            key_points = question.get("key_points", [])
            if len(key_points) < 2:
                issues.append(f"! Недостаточно ключевых моментов ({len(key_points)} вместо минимум 2)")

            if len(set(key_points)) != len(key_points):
                issues.append("! Есть дублирующиеся ключевые моменты")

            q_len = len(question.get("question", "").split())
            if q_len < 5:
                issues.append(f"! Вопрос слишком короткий ({q_len} слов)")

        explanation = question.get("explanation") or question.get("sample_answer") or ""
        if len(explanation.split()) < 10:
            issues.append("! Объяснение слишком краткое")

        return issues

    def _generate_recommendations(
        self,
        q_type: str,
        question: dict,
        metrics: Dict[str, MetricScore],
        issues: List[str]
    ) -> List[str]:
        """Генерация рекомендаций на основе метрик"""
        recommendations = []

        if metrics.get("Relevance", MetricScore(name="", score=0.5, reason="")).score < 0.6:
            recommendations.append(" Улучшите связь вопроса с контентом курса")

        if metrics.get("Coherence", MetricScore(name="", score=0.5, reason="")).score < 0.6:
            recommendations.append(" Переформулируйте вопрос для большей ясности")

        if metrics.get("Faithfulness", MetricScore(name="", score=0.5, reason="")).score < 0.6:
            recommendations.append(" Проверьте корректность ответа и соответствие контенту")

        if metrics.get("Conciseness", MetricScore(name="", score=0.5, reason="")).score < 0.4:
            recommendations.append(" Оптимизируйте длину вопроса")

        if len(issues) > 0:
            recommendations.append(f" Исправьте {len(issues)} структурных проблем")

        return recommendations

    async def evaluate_question(
        self,
        question: dict,
        q_type: str,
        q_index: int,
        content: str
    ) -> QuestionEvaluation:
        """Полная оценка одного вопроса с GEval метриками"""
        print(f"  -> Оценка {q_type} вопроса #{q_index + 1}...")

        # Получаем текст вопроса
        if q_type == "multiple_choice":
            q_text = question.get("question", "")
            expected = question.get("explanation", "")
        elif q_type == "true_false":
            q_text = question.get("statement", "")
            expected = question.get("explanation", "")
        else:  # open_ended
            q_text = question.get("question", "")
            expected = question.get("sample_answer", "")

        # Оцениваем по всем метрикам параллельно
        relevance_score, coherence_score, conciseness_score, faithfulness_score = await asyncio.gather(
            self.relevance_metric.evaluate(q_text, content, expected, self.semaphore),
            self.coherence_metric.evaluate(q_text, content, expected, self.semaphore),
            self.conciseness_metric.evaluate(q_text, content, expected, self.semaphore),
            self.faithfulness_metric.evaluate(q_text, content, expected, self.semaphore)
        )

        # Собираем все метрики
        metrics = {
            "Relevance": relevance_score,
            "Coherence": coherence_score,
            "Conciseness": conciseness_score,
            "Faithfulness": faithfulness_score
        }

        # Структурные проблемы
        issues = self._check_structural_issues(question, q_type)

        # Рекомендации
        recommendations = self._generate_recommendations(q_type, question, metrics, issues)

        # Общий скор (взвешенное среднее)
        overall_score = (
            relevance_score.score * 0.3 +      # 30%
            coherence_score.score * 0.25 +     # 25%
            faithfulness_score.score * 0.25 +  # 25%
            conciseness_score.score * 0.2      # 20%
        ) * 100

        return QuestionEvaluation(
            question_type=q_type,
            question_id=q_index,
            relevance_metric=relevance_score,
            coherence_metric=coherence_score,
            conciseness_metric=conciseness_score,
            faithfulness_metric=faithfulness_score,
            issues=issues,
            overall_score=overall_score,
            recommendations=recommendations
        )

    async def evaluate_topic(
        self,
        topic_name: str,
        questions: dict,
        content: str
    ) -> TopicEvaluation:
        """Оценка всех вопросов по теме"""
        print(f"\n>> Оценка темы: '{topic_name}'")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        evaluations = []
        q_index = 0

        # Оцениваем MC вопросы
        for q in questions.get("multiple_choice", []):
            eval_result = await self.evaluate_question(
                q, "multiple_choice", q_index, content
            )
            evaluations.append(eval_result)
            q_index += 1

        # Оцениваем T/F вопросы
        for q in questions.get("true_false", []):
            eval_result = await self.evaluate_question(
                q, "true_false", q_index, content
            )
            evaluations.append(eval_result)
            q_index += 1

        # Оцениваем Open-Ended вопросы
        for q in questions.get("open_ended", []):
            eval_result = await self.evaluate_question(
                q, "open_ended", q_index, content
            )
            evaluations.append(eval_result)
            q_index += 1

        # Вычисляем средние значения
        if evaluations:
            avg_relevance = sum(e.relevance_metric.score for e in evaluations) / len(evaluations)
            avg_coherence = sum(e.coherence_metric.score for e in evaluations) / len(evaluations)
            avg_conciseness = sum(e.conciseness_metric.score for e in evaluations) / len(evaluations)
            avg_faithfulness = sum(e.faithfulness_metric.score for e in evaluations) / len(evaluations)
            overall_avg = sum(e.overall_score for e in evaluations) / len(evaluations)
        else:
            avg_relevance = avg_coherence = avg_conciseness = avg_faithfulness = overall_avg = 0

        topic_eval = TopicEvaluation(
            topic=topic_name,
            total_questions=len(evaluations),
            multiple_choice_count=len(questions.get("multiple_choice", [])),
            true_false_count=len(questions.get("true_false", [])),
            open_ended_count=len(questions.get("open_ended", [])),
            evaluations=evaluations,
            avg_relevance=avg_relevance,
            avg_coherence=avg_coherence,
            avg_conciseness=avg_conciseness,
            avg_faithfulness=avg_faithfulness,
            overall_avg_score=overall_avg
        )

        print(f"   Средний счёт темы: {overall_avg:.1f}/100")

        return topic_eval

    def _calculate_distribution(self, scores: List[float]) -> Dict[str, float]:
        """Вычисление распределения оценок"""
        if not scores:
            return {"excellent": 0, "good": 0, "fair": 0, "poor": 0}

        excellent = sum(1 for s in scores if s >= 85) / len(scores) * 100
        good = sum(1 for s in scores if 70 <= s < 85) / len(scores) * 100
        fair = sum(1 for s in scores if 55 <= s < 70) / len(scores) * 100
        poor = sum(1 for s in scores if s < 55) / len(scores) * 100

        return {
            "excellent": round(excellent, 1),
            "good": round(good, 1),
            "fair": round(fair, 1),
            "poor": round(poor, 1)
        }

    async def evaluate_course(
        self,
        course_data: dict,
        questions_data: dict
    ) -> CourseEvaluation:
        """Полная оценка всего курса"""
        print(f"\n{'='*60}")
        print(f" ОЦЕНКА КУРСА: {course_data['course_title']}")
        print(f"{'='*60}")

        topic_evaluations = []
        all_scores = []
        critical_issues = []

        # Обходим все главы и подтемы
        for chapter in course_data.get("chapters", []):
            chapter_title = chapter.get("title", "")
            chapter_content = chapter.get("content", "")

            # Ищем соответствующие вопросы в results
            chapter_questions = next(
                (c for c in questions_data.get("chapters", [])
                 if c.get("title") == chapter_title),
                None
            )

            if not chapter_questions:
                continue

            # Оцениваем подтемы
            for sub_result in chapter_questions.get("sub_topics", []):
                subtopic_title = sub_result.get("title", "")
                subtopic_content = next(
                    (st.get("content", "")
                     for st in chapter.get("sub_topics", [])
                     if st.get("title") == subtopic_title),
                    ""
                )

                # Получаем вопросы
                questions_dict = sub_result.get("questions", {})

                # Оцениваем тему
                topic_eval = await self.evaluate_topic(
                    subtopic_title,
                    questions_dict,
                    subtopic_content or chapter_content
                )

                topic_evaluations.append(topic_eval)

                # Собираем скоры
                for eval_result in topic_eval.evaluations:
                    all_scores.append(eval_result.overall_score)

                    # Ищем критические проблемы
                    if eval_result.overall_score < 50:
                        critical_issues.append(
                            f" Критическая проблема в {topic_eval.topic} "
                            f"(вопрос #{eval_result.question_id + 1}): счёт {eval_result.overall_score:.0f}/100"
                        )

        # Вычисляем финальные метрики
        total_questions = sum(t.total_questions for t in topic_evaluations)
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0

        # Определяем рейтинг
        if overall_score >= 85:
            rating = "Excellent ⭐⭐⭐⭐⭐"
        elif overall_score >= 70:
            rating = "Good ⭐⭐⭐⭐"
        elif overall_score >= 55:
            rating = "Fair ⭐⭐⭐"
        else:
            rating = "Poor ⭐⭐"

        distribution = self._calculate_distribution(all_scores)

        evaluation = CourseEvaluation(
            course_title=course_data["course_title"],
            total_topics=len(topic_evaluations),
            total_questions=total_questions,
            topic_evaluations=topic_evaluations,
            overall_score=round(overall_score, 1),
            quality_rating=rating,
            distribution=distribution,
            critical_issues=critical_issues[:10]
        )

        return evaluation

    def save_evaluation(
        self,
        evaluation: CourseEvaluation,
        output_file: str = "evaluation_results_geval.json"
    ) -> str:
        """Сохранение результатов оценки"""
        with open(output_file, 'w', encoding='utf-8') as f:
            # Конвертируем Pydantic модели в dict
            json.dump(
                json.loads(evaluation.model_dump_json()),
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"\n Результаты сохранены в {output_file}")
        return output_file

    def print_evaluation_summary(self, evaluation: CourseEvaluation):
        """Красивая печать результатов"""
        print(f"\n{'='*60}")
        print(f">> ИТОГОВЫЙ ОТЧЕТ")
        print(f"{'='*60}")
        print(f"Курс: {evaluation.course_title}")
        print(f"Всего тем: {evaluation.total_topics}")
        print(f"Всего вопросов: {evaluation.total_questions}")
        print(f"\n>> Общий счёт: {evaluation.overall_score}/100")
        print(f"Рейтинг: {evaluation.quality_rating}")

        print(f"\n>> Распределение оценок:")
        for level, percentage in evaluation.distribution.items():
            print(f"  {level.capitalize()}: {percentage}%")

        if evaluation.critical_issues:
            print(f"\n>> Критические проблемы ({len(evaluation.critical_issues)}):")
            for issue in evaluation.critical_issues[:5]:
                print(f"  {issue}")

        print(f"\n{'='*60}\n")

    def print_detailed_metrics(self, evaluation: CourseEvaluation):
        """Детальный отчёт по метрикам для каждой темы"""
        print(f"\n{'='*60}")
        print(f">> ДЕТАЛЬНЫЙ АНАЛИЗ МЕТРИК")
        print(f"{'='*60}\n")

        for topic in evaluation.topic_evaluations:
            print(f">> Тема: {topic.topic}")
            print(f"   Всего вопросов: {topic.total_questions}")
            print(f"   ├─ Multiple Choice: {topic.multiple_choice_count}")
            print(f"   ├─ True/False: {topic.true_false_count}")
            print(f"   └─ Open-Ended: {topic.open_ended_count}")

            print(f"\n   Средние оценки по метрикам:")
            print(f"   ├─ Relevance (Релевантность):   {topic.avg_relevance:.2f}/1.0")
            print(f"   ├─ Coherence (Связность):      {topic.avg_coherence:.2f}/1.0")
            print(f"   ├─ Conciseness (Лаконичность): {topic.avg_conciseness:.2f}/1.0")
            print(f"   └─ Faithfulness (Верность):    {topic.avg_faithfulness:.2f}/1.0")
            print(f"\n   Общий скор темы: {topic.overall_avg_score:.1f}/100\n")
