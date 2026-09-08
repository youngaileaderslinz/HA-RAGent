import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import httpx
from openai import AsyncOpenAI, APIConnectionError
import pytest

from custom_components.ha_ragent.src import const
from custom_components.ha_ragent.src.backends.llm import ollama_backend as ollama_llm
from custom_components.ha_ragent.src.backends.embedder import ollama_backend as ollama_embedder
from custom_components.ha_ragent.src.backends.llm.openai_backend import OpenAiLlmBackend
from custom_components.ha_ragent.src.backends.embedder.openai_backend import OpenAiEmbedder


CONFIG = {
    const.CONF_LLM_MODEL: "test", const.CONF_EMBEDDING_MODEL: "test",
    const.CONF_ENABLE_MODEL_THINKING: False, const.CONF_TEMPERATURE: 0.1,
    const.CONF_MAX_TOKENS: 100, const.CONF_CONTEXT_LENGTH: 4096,
}
MESSAGES = [{"role": "user", "content": "turn on the light"}]


class Response:
    def __init__(self, data=None, lines=(), error=None):
        self.data = data
        self.lines = lines
        self.error = error
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        self.closed = True

    def raise_for_status(self):
        if self.error:
            raise self.error

    async def json(self):
        if isinstance(self.data, BaseException):
            raise self.data
        return self.data

    @property
    def content(self):
        async def chunks():
            for line in self.lines:
                if isinstance(line, BaseException):
                    raise line
                yield json.dumps(line).encode() + b"\n"
        return chunks()


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, deepcopy(kwargs)))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


@pytest.fixture
def backoff(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(ollama_llm.asyncio, "sleep", sleep)
    return sleep


def ollama_backend(monkeypatch, session, embedding=False):
    module = ollama_embedder if embedding else ollama_llm
    monkeypatch.setattr(module, "async_get_clientsession", lambda _hass: session)
    cls = module.OllamaEmbedder if embedding else module.OllamaLlmBackend
    return cls(SimpleNamespace(), {})


async def chat(backend):
    return [chunk async for chunk in backend.async_send_chat_request(CONFIG, MESSAGES, [])]


@pytest.mark.parametrize("error", [aiohttp.ClientConnectionError(), TimeoutError(),
                                   aiohttp.ClientPayloadError()])
def test_ollama_reconnects_before_output_without_truncating_or_enabling_thinking(monkeypatch, backoff, error):
    failed = Response(lines=[error])
    success = Response(lines=[{"message": {"content": "done"}, "done": True}])
    session = Session(failed, success)
    backend = ollama_backend(monkeypatch, session)

    assert asyncio.run(chat(backend)) == ["done"]
    assert len(session.requests) == 2
    assert session.requests[0] == session.requests[1]
    assert session.requests[1][2]["json"]["think"] is False
    assert session.requests[1][2]["json"]["messages"] == MESSAGES
    assert failed.closed and success.closed
    backoff.assert_awaited_once_with(0.5)


@pytest.mark.parametrize("message", [
    {"content": "partial answer"},
    {"tool_calls": [{"function": {"name": "HassTurnOn", "arguments": {"name": "light"}}}]},
])
def test_ollama_does_not_replay_partial_text_or_tool_calls(monkeypatch, backoff, message):
    response = Response(lines=[{"message": message}, aiohttp.ClientConnectionError()])
    session = Session(response)
    backend = ollama_backend(monkeypatch, session)
    emitted = []

    async def run():
        with pytest.raises(aiohttp.ClientConnectionError):
            async for chunk in backend.async_send_chat_request(CONFIG, MESSAGES, []):
                emitted.append(chunk)

    asyncio.run(run())
    assert len(emitted) == 1
    assert len(session.requests) == 1
    assert response.closed
    backoff.assert_not_awaited()


def test_ollama_connection_retries_are_bounded(monkeypatch, backoff):
    attempts = const.CONNECTION_RETRIES + 1
    session = Session(*(aiohttp.ClientConnectionError() for _ in range(attempts)))
    backend = ollama_backend(monkeypatch, session)

    with pytest.raises(aiohttp.ClientConnectionError):
        asyncio.run(chat(backend))

    assert len(session.requests) == attempts
    assert [call.args[0] for call in backoff.await_args_list] == [
        const.RETRY_BACKOFF_BASE_SECONDS
        * (const.RETRY_BACKOFF_MULTIPLIER**attempt)
        for attempt in range(const.CONNECTION_RETRIES)
    ]


@pytest.mark.parametrize("status,retries", [(400, 0), (401, 0), (403, 0), (404, 0),
                                          (429, 1), (503, 1)])
def test_ollama_retries_only_transient_http_statuses(monkeypatch, backoff, status, retries):
    error = aiohttp.ClientResponseError(SimpleNamespace(real_url="http://test"), (), status=status)
    session = Session(Response(error=error), Response(lines=[{"message": {"content": "ok"}}]))
    backend = ollama_backend(monkeypatch, session)

    if retries:
        assert asyncio.run(chat(backend)) == ["ok"]
    else:
        with pytest.raises(aiohttp.ClientResponseError):
            asyncio.run(chat(backend))
    assert len(session.requests) == 1 + retries
    assert backoff.await_count == retries


def test_ollama_cancellation_propagates_without_retry(monkeypatch, backoff):
    response = Response(lines=[asyncio.CancelledError()])
    session = Session(response)
    backend = ollama_backend(monkeypatch, session)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(chat(backend))
    assert response.closed
    assert len(session.requests) == 1
    backoff.assert_not_awaited()


def test_ollama_closes_connection_when_consumer_stops(monkeypatch, backoff):
    response = Response(lines=[{"message": {"content": "partial"}}, aiohttp.ClientConnectionError()])
    backend = ollama_backend(monkeypatch, Session(response))

    async def run():
        stream = backend.async_send_chat_request(CONFIG, MESSAGES, [])
        assert await anext(stream) == "partial"
        await stream.aclose()
        assert response.closed

    asyncio.run(run())
    backoff.assert_not_awaited()


def test_embedding_retries_whole_batch_after_truncated_response(monkeypatch, backoff):
    failed = Response(data=aiohttp.ClientPayloadError())
    session = Session(failed, Response(data={"embeddings": [[1.0], [2.0]]}))
    backend = ollama_backend(monkeypatch, session, embedding=True)

    assert asyncio.run(backend._async_embed_batch(CONFIG, ["first", "second"])) == [[1.0], [2.0]]
    assert session.requests[0] == session.requests[1]
    assert session.requests[0][2]["json"]["input"] == ["first", "second"]
    assert session.requests[0][2]["timeout"].sock_read == const.STREAM_READ_TIMEOUT
    assert failed.closed


@pytest.mark.parametrize("embedding", [False, True])
def test_model_discovery_recovers_connection_failure(monkeypatch, backoff, embedding):
    session = Session(aiohttp.ClientConnectionError(), Response(data={"models": [{"name": "test"}]}),
                      Response(data={"capabilities": ["tools", "embedding"]}))
    backend = ollama_backend(monkeypatch, session, embedding=embedding)

    assert asyncio.run(backend.async_get_available_models()) == ["test"]
    assert len(session.requests) == 3
    backoff.assert_awaited_once_with(0.5)


@pytest.mark.parametrize("embedding", [False, True])
def test_openai_sdk_recovers_connection_failure_with_original_payload(backoff, embedding):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            raise httpx.ConnectError("disconnected", request=request)
        if embedding:
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
        chunk = {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test",
                 "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]}
        return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            async with AsyncOpenAI(api_key="test", http_client=transport,
                                   max_retries=const.CONNECTION_RETRIES) as client:
                cls = OpenAiEmbedder if embedding else OpenAiLlmBackend
                backend = cls(SimpleNamespace(), {})
                backend._client = client
                return await backend.async_embed_text(CONFIG, "test") if embedding else await chat(backend)

    assert asyncio.run(run()) == ([1.0] if embedding else ["ok"])
    assert len(requests) == 2
    assert requests[0] == requests[1]
    if not embedding:
        assert requests[1]["chat_template_kwargs"]["enable_thinking"] is False


def test_openai_sdk_connection_retries_are_not_multiplied(backoff):
    attempts = []

    def handler(request):
        attempts.append(request)
        raise httpx.ConnectError("offline", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            async with AsyncOpenAI(api_key="test", http_client=transport,
                                   max_retries=const.CONNECTION_RETRIES) as client:
                backend = OpenAiLlmBackend(SimpleNamespace(), {})
                backend._client = client
                with pytest.raises(APIConnectionError):
                    await chat(backend)

    asyncio.run(run())
    assert len(attempts) == const.CONNECTION_RETRIES + 1


def test_openai_does_not_replay_and_closes_failed_stream(backoff):
    attempts = []

    class BrokenStream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            chunk = {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "test",
                     "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]}
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            raise httpx.ReadError("connection lost")

        async def aclose(self):
            self.closed = True

    response_stream = BrokenStream()

    def handler(request):
        attempts.append(request)
        return httpx.Response(200, stream=response_stream, headers={"content-type": "text/event-stream"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            async with AsyncOpenAI(api_key="test", http_client=transport,
                                   max_retries=const.CONNECTION_RETRIES) as client:
                backend = OpenAiLlmBackend(SimpleNamespace(), {})
                backend._client = client
                emitted = []
                with pytest.raises(httpx.ReadError):
                    async for chunk in backend.async_send_chat_request(CONFIG, MESSAGES, []):
                        emitted.append(chunk)
                assert emitted == ["partial"]
                assert response_stream.closed

    asyncio.run(run())
    assert len(attempts) == 1
