import asyncio
import json
from typing import Optional

from gptcache.utils import import_httpx

import_httpx()

import httpx  # pylint: disable=C0413


_CLIENT_HEADER = {"Content-Type": "application/json", "Accept": "application/json"}


class Client:
    """GPTCache client to send requests to GPTCache server.

    :param uri: the uri leads to the server, defaults to "http://localhost:8000".
    :type uri: str

    Example:
        .. code-block:: python

            from gptcache import client

            client = Client(uri="http://localhost:8000")
            client.put("Hi", "Hi back")
            ans = client.get("Hi")
    """

    def __init__(self, uri: str = "http://localhost:8000"):
        self._uri = uri

    async def _put(
        self, question: str, answer: str, application_id: Optional[str] = None
    ):
        async with httpx.AsyncClient() as client:
            data = {
                "prompt": question,
                "answer": answer,
                "application_id": application_id,
            }

            response = await client.post(
                f"{self._uri}/put", headers=_CLIENT_HEADER, data=json.dumps(data)
            )

        return response.status_code

    async def _get(self, question: str, application_id: Optional[str] = None):
        async with httpx.AsyncClient() as client:
            data = {
                "prompt": question,
                "application_id": application_id,
            }

            response = await client.post(
                f"{self._uri}/get", headers=_CLIENT_HEADER, data=json.dumps(data)
            )

        return response.json().get("answer")

    def put(
        self, question: str, answer: str, application_id: Optional[str] = None
    ):
        """
        :param question: the question to be put.
        :type question: str
        :param answer: the answer to the question to be put.
        :type answer: str
        :param application_id: optional application id from register_application_slo (JSON null if omitted).
        :type application_id: Optional[str]
        :return: status code.
        """
        return asyncio.run(self._put(question, answer, application_id))

    def get(self, question: str, application_id: Optional[str] = None):
        """
        :param question: the question to get an answer.
        :type question: str
        :param application_id: optional application id for scoped cache behavior.
        :type application_id: Optional[str]
        :return: answer to the question.
        """
        return asyncio.run(self._get(question, application_id))

    async def _flush(self):
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{self._uri}/flush", headers=_CLIENT_HEADER)
        return response.status_code, response.text

    def flush(self):
        """
        Flush cache storage on the server.

        :return: (HTTP status code, response body text).
        """
        return asyncio.run(self._flush())

    async def _register_application_slo(
        self, latency_p99_ms: float, accuracy_slo: float
    ) -> str:
        async with httpx.AsyncClient() as client:
            data = {
                "latency_p99_ms": latency_p99_ms,
                "accuracy_slo": accuracy_slo,
            }
            response = await client.post(
                f"{self._uri}/v1/applications",
                headers=_CLIENT_HEADER,
                data=json.dumps(data),
            )
            response.raise_for_status()
            return str(response.json().get("application_id", ""))

    def register_application_slo(self, latency_p99_ms: float, accuracy_slo: float) -> str:
        """
        Register latency (p99 ms) and accuracy SLOs with the server.

        :param latency_p99_ms: target p99 latency in milliseconds.
        :param accuracy_slo: minimum accuracy in [0.0, 1.0].
        :return: server-issued application_id.
        """
        return asyncio.run(self._register_application_slo(latency_p99_ms, accuracy_slo))

    async def _deregister_application(self, application_id: str) -> None:
        aid = (application_id or "").strip()
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self._uri}/v1/applications/{aid}",
                headers=_CLIENT_HEADER,
            )
        response.raise_for_status()

    def deregister_application(self, application_id: str) -> None:
        """
        Remove a registered application from the server by ``application_id``.

        :raises httpx.HTTPStatusError: if the id is unknown (404) or the request fails.
        """
        return asyncio.run(self._deregister_application(application_id))
