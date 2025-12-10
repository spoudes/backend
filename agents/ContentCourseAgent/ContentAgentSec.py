from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class SecurityAgent:
    def __init__(self, llm):
        """
        llm: Модель LangChain (например, self.text_llm из основного класса), 
             желательно быстрая и дешевая (Mistral Small / Haiku).
        """
        self.llm = llm
        
        # Промпт для проверки безопасности
        # Мы просим модель вернуть только 'SAFE' или 'UNSAFE'.
        self.prompt = PromptTemplate(
            template="""
            Ты - AI-модератор контента. Твоя задача - проверить следующий текст на наличие:
            1. Вредоносного кода (malware/virus/exploit).
            2. Инструкций по созданию оружия или наркотиков.
            3. Разжигания ненависти, экстремизма или дискриминации.
            4. Конфиденциальных данных (PII), если они выглядят настоящими.

            ТЕКСТ ДЛЯ ПРОВЕРКИ:
            ---
            {text}
            ---

            Если текст безопасен и пригоден для образовательных целей, ответь одним словом: SAFE.
            Если обнаружено нарушение, ответь одним словом: UNSAFE.
            """,
            input_variables=["text"]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    async def check_context(self, text: str) -> bool:
        """
        Возвращает True, если контекст безопасен.
        """
        if not text or len(text.strip()) < 10:
            return True # Слишком короткий текст считаем безопасным или пустым
            
        try:
            # Ограничиваем длину проверяемого текста, чтобы не тратить токены (например, первые 4000 символов)
            response = await self.chain.ainvoke({"text": text[:4000]})
            clean_resp = response.strip().upper()
            
            if "UNSAFE" in clean_resp:
                print(f"!!! SECURITY ALERT: Контекст помечен как небезопасный !!!")
                return False
            return True
            
        except Exception as e:
            print(f"Ошибка Security Agent: {e}. Пропускаем проверку (fail-open).")
            # В зависимости от требований безопасности, здесь можно вернуть False (fail-closed)
            return True
