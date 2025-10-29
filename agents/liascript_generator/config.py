from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(alias="OPENAI_API_KEY", default="")
    openai_model: str = Field(alias="OPENAI_MODEL", default="gpt-4o-mini")
    llm_temperature: float = Field(alias="LLM_TEMPERATURE", default=0.2)
    google_api_key: str = Field(alias="GOOGLE_API_KEY", default="")
    google_model: str = Field(alias="GOOGLE_MODEL", default="gemini-2.5-flash")
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()