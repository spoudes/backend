import time
from http import HTTPStatus

import pytest
from httpx import AsyncClient

from agents.liascript_generator.validate_and_cleanify import validate_liascript
from tests.conftest import BASE_URL, TEST_FOLDER_ID

@pytest.mark.anyio
async def test_pipeline(client: AsyncClient):
    start_time = time.perf_counter()
    response = await client.get(f"{BASE_URL}/generate-course/{TEST_FOLDER_ID}")
    is_valid, _ = validate_liascript(response.text)

    elapsed_time = time.perf_counter() - start_time

    print(f"Время выполнения ручки: {elapsed_time:.4f} сек")
    assert response.status_code == HTTPStatus.OK
    assert is_valid