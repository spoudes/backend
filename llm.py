from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import settings


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> BaseChatModel:
    llm = ChatOpenAI(
        api_key=settings.openai_api_key or None,
        model=model or settings.openai_model,
        temperature=temperature
        if temperature is not None
        else settings.llm_temperature,
        timeout=timeout or settings.timeout_s,
        base_url=settings.openai_base_url or None,  # совместимые провайдеры
    )
    return llm


def simple_chain(system_msg: str = "You are a helpful assistant."):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_msg), ("human", "{input}")]
    )
    return prompt | get_llm() | StrOutputParser()