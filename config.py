from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(alias="OPENAI_API_KEY", default="")
    openai_model: str = Field(alias="OPENAI_MODEL", default="gpt-4o-mini")
    openai_base_url: str = Field(alias="OPENAI_BASE_URL", default="")
    llm_temperature: float = Field(alias="LLM_TEMPERATURE", default=0.2)
    timeout_s: int = Field(alias="TIMEOUT_S", default=60)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()