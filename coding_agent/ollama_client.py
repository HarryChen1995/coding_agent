"""Talks to Ollama directly over HTTP (/api/chat) — no `ollama` python
package required, just httpx. Mirrors the message shape the rest of the
agent already expects: chat() returns the `message` dict with
role/content/tool_calls, same as the ollama library did.
"""

import os

import httpx

DEFAULT_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_API_KEY = os.environ.get("OLLAMA_API_KEY", "")  # set via env, never hardcode


class OllamaError(Exception):
    pass


async def chat(
    model: str,
    messages: list,
    tools: list = None,
    format: str = None,
    base_url: str = None,
    api_key: str = None,
    timeout: float = 120.0,
) -> dict:
    """POST /api/v1/chat/completions (OpenAI-compatible) with stream=false
    and return the `message` dict.

    This hits our Open WebUI gateway in front of Ollama, not Ollama's native
    /api/chat — the gateway only exposes the OpenAI-style route. Response
    shape is `choices[0].message`, but that dict already has the
    role/content/tool_calls keys the rest of the agent expects, same as
    Ollama's native `message` field would.

    If an API key is set (via `api_key`, falling back to $OLLAMA_API_KEY),
    it's sent as `Authorization: Bearer <key>`.

    Raises OllamaError on a non-2xx response, a connection failure, or an
    unexpected response shape — callers (agent.py, intent.py) already retry
    on this.
    """
    url = f"{(base_url or DEFAULT_BASE_URL).rstrip('/')}/api/v1/chat/completions"
    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    if format == "json":
        payload["response_format"] = {"type": "json_object"}

    key = api_key or DEFAULT_API_KEY
    headers = {"Authorization": f"Bearer {key}"} if key else {}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        body = e.response.text[:300]
        raise OllamaError(f"Ollama API returned {e.response.status_code} for {model}: {body}") from e
    except httpx.RequestError as e:
        raise OllamaError(
            f"Could not reach Ollama at {url} ({e}). Is `ollama serve` running?"
        ) from e

    choices = data.get("choices")
    if not choices:
        raise OllamaError(f"Unexpected response shape from Ollama (no 'choices' key): {data}")
    message = choices[0].get("message")
    if message is None:
        raise OllamaError(f"Unexpected response shape from Ollama (no 'message' key): {data}")
    return message
