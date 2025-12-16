import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from main import app

BASE_URL = "" #TODO: вынести в энв
TEST_FOLDER_ID = "4012fcde-e79b-4fa1-b58e-489182364643"

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
async def app_instance():
    """Запускаем приложение один раз на всю сессию тестов"""
    async with LifespanManager(app) as manager:
        yield manager.app


@pytest.fixture(scope="function")
async def client(app_instance):
    """Легкий клиент, пересоздается для каждого теста"""
    async with AsyncClient(transport=ASGITransport(app_instance), base_url="http://testserver") as c:
        yield c