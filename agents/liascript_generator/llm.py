from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .config import settings
from dotenv import load_dotenv

load_dotenv()

def get_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    llm = ChatGoogleGenerativeAI(
        api_key=settings.google_api_key or None,
        model=model or settings.openai_model,
        temperature=temperature
        if temperature is not None
        else settings.llm_temperature,  
    )
    return llm


def simple_chain(system_msg: str = "You are a helpful assistant."):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_msg), ("human", "{input}")]
    )
    llm = ChatMistralAI(api_key=settings.mistral_api_key, temperature=0, model_name="mistral-medium-2508")
    return prompt | llm | StrOutputParser()