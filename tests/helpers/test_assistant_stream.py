import asyncio

import pytest

from custom_components.ha_ragent.src.homeassistant.helpers.assistant_stream import AssistantStream


TOOL_CALL = '```homeassistant\n{"tool":"HassTurnOn","arguments":{"name":"light.desk"}}\n```'


async def collect(chunks):
    async def source():
        for chunk in chunks:
            yield chunk

    stream = AssistantStream(source())
    deltas = [delta async for delta in stream.async_deltas()]
    assert stream.content == "".join(delta["content"] for delta in deltas)
    return stream.response, deltas


@pytest.mark.parametrize("split", range(len(TOOL_CALL) + 1))
def test_split_tool_markup_is_hidden_and_original_call_is_retained(split):
    response, deltas = asyncio.run(collect([
        "  Please wait.\n", TOOL_CALL[:split], TOOL_CALL[split:], "\nDone.  ",
    ]))

    assert response == "  Please wait.\n" + TOOL_CALL + "\nDone.  "
    assert "".join(delta["content"] for delta in deltas) == "Please wait.\n\nDone."
    assert deltas[0]["role"] == "assistant"
    assert all(set(delta) == {"content"} for delta in deltas[1:])
    assert all("tool_calls" not in delta for delta in deltas)


@pytest.mark.parametrize("chunks,expected", [
    (["The light", " is on."], "The light is on."),
    (list("  A word at a time.  "), "A word at a time."),
    ([TOOL_CALL], ""),
    (list(TOOL_CALL), ""),
    ([TOOL_CALL, TOOL_CALL, "Done."], "Done."),
    (["  \n", " \t"], ""),
    (["Use `", "name", "`"], "Use `name`"),
    (["```json\n", '{"name":"light.desk"}', "\n```"], '```json\n{"name":"light.desk"}\n```'),
    (["Text", "```homeassistant\n", '{"tool":"incomplete'], "Text"),
    (["Text", "```homeassist"], "Text"),
])
def test_stream_keeps_speech_and_ordinary_markdown(chunks, expected):
    response, deltas = asyncio.run(collect(chunks))

    assert response == "".join(chunks)
    assert "".join(delta["content"] for delta in deltas) == expected


def test_first_delta_arrives_before_generation_finishes():
    async def check():
        closed = False

        async def source():
            nonlocal closed
            try:
                yield "The light"
                yield " is on."
            finally:
                closed = True

        stream = AssistantStream(source())
        deltas = stream.async_deltas()
        assert await anext(deltas) == {"role": "assistant", "content": "The light"}
        assert stream.response == "The light"
        assert not closed
        await deltas.aclose()
        assert closed

    asyncio.run(check())


def test_stream_failure_propagates_without_exposing_partial_tool_arguments():
    async def check():
        async def source():
            yield "```homeassistant\n{\"tool\":"
            raise ConnectionError("stream lost")

        stream = AssistantStream(source())
        deltas = []
        with pytest.raises(ConnectionError, match="stream lost"):
            async for delta in stream.async_deltas():
                deltas.append(delta)
        assert deltas == []
        assert stream.response == "```homeassistant\n{\"tool\":"

    asyncio.run(check())
