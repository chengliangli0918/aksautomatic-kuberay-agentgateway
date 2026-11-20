"""Request rules for the quota-limited chat API, kept free of Streamlit so they can be tested."""

from __future__ import annotations

import base64
import io
import ipaddress
import math
import unicodedata
import urllib.parse
from collections.abc import Mapping

import openai
from PIL import Image, ImageOps

# These must match the deployment: vLLM max_model_len in deploy/rayservice.yaml and the
# limits enforced by cmd/quota-proxy/main.go and cmd/quota-proxy/models.go.
CONTEXT_TOKENS = 8192
# gpt-oss reasons before answering, and its reasoning tokens count toward this limit.
MAX_OUTPUT_TOKENS = 4096
MAX_MESSAGES = 128
# Both tokenizers are byte-level BPE, so no token is shorter than one byte. Qwen3.5 first
# NFC-normalizes the text, which lengthens some characters, such as U+1D160; gpt-oss's
# o200k tokenizer encodes the raw text. The larger of the two UTF-8 byte counts bounds
# both. The allowances exceed the chat templates' overhead, including the system message
# that gpt-oss's template always adds. A request that overflows the context still costs a
# full quota reservation, so the client never sends one.
PER_MESSAGE_TOKENS = 8
FIXED_TOKENS = 128
# The engine resizes each image to at most MAX_IMAGE_PIXELS and turns every 32x32 block into
# one token, plus 2 tokens that mark the image. The proxy accepts at most MAX_IMAGES PNG or
# JPEG images of at most MAX_IMAGE_BYTES per request.
MAX_IMAGES = 2
MAX_IMAGE_PIXELS = 2_097_152
IMAGE_TOKENS = MAX_IMAGE_PIXELS // (32 * 32) + 2
MAX_IMAGE_BYTES = 4 * 1024 * 1024


def is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def api_base_url(endpoint: str) -> str:
    """Return the OpenAI base URL (ending in /v1) for an endpoint such as https://host."""
    url = endpoint.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")]
    parts = urllib.parse.urlsplit(url)
    if not parts.hostname or parts.path or parts.query or parts.fragment or parts.username:
        raise ValueError("Enter the endpoint as https://<hostname>.")
    if parts.scheme != "https" and not (parts.scheme == "http" and is_loopback(parts.hostname)):
        raise ValueError("Use https://; API keys must never be sent over plain HTTP.")
    return url + "/v1"


def image_data_url(data: bytes) -> str:
    """Return an uploaded image as a PNG or JPEG data URL that the quota proxy accepts.

    The image is turned upright and scaled down to the engine's pixel limit, so the upload is
    no larger than what the model sees.
    """
    try:
        with Image.open(io.BytesIO(data)) as source:
            source_format = source.format
            image = ImageOps.exif_transpose(source)
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError("Attach a PNG, JPEG, WebP, GIF, or BMP image.") from error
    width, height = image.size
    if max(width, height) > 200 * min(width, height):
        raise ValueError("The image is too narrow: its aspect ratio must be at most 200:1.")
    if width * height > MAX_IMAGE_PIXELS:
        scale = math.sqrt(MAX_IMAGE_PIXELS / (width * height))
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    image = image.convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB")
    # PNG keeps screenshots sharp; JPEG keeps photos small.
    for image_format in ("JPEG",) if source_format == "JPEG" else ("PNG", "JPEG"):
        buffer = io.BytesIO()
        if image_format == "JPEG":
            flat = Image.new("RGB", image.size, "white")
            flat.paste(image, mask=image.getchannel("A") if image.mode == "RGBA" else None)
            flat.save(buffer, "JPEG", quality=90)
        else:
            image.save(buffer, "PNG")
        if buffer.tell() <= MAX_IMAGE_BYTES:
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:image/{image_format.lower()};base64,{encoded}"
    raise ValueError("The image is too large, even after scaling it down.")


def request_content(message: dict) -> str | list[dict]:
    """Return the API content of a history message: its text, or image parts and its text."""
    images = message.get("images") or []
    if not images:
        return message["content"]
    parts = [{"type": "image_url", "image_url": {"url": url}} for url in images]
    if message["content"].strip():
        parts.append({"type": "text", "text": message["content"]})
    return parts


def _parts(message: dict) -> list[dict]:
    content = message["content"]
    return content if isinstance(content, list) else [{"type": "text", "text": content}]


def image_count(messages: list[dict]) -> int:
    return sum(p["type"] == "image_url" for m in messages for p in _parts(m))


def text_tokens(text: str) -> int:
    return max(len(text.encode("utf-8")), len(unicodedata.normalize("NFC", text).encode("utf-8")))


def token_upper_bound(messages: list[dict]) -> int:
    return FIXED_TOKENS + sum(
        PER_MESSAGE_TOKENS
        + sum(IMAGE_TOKENS if p["type"] == "image_url" else text_tokens(p["text"]) for p in _parts(m))
        for m in messages
    )


def request_messages(system_prompt: str, history: list[dict], max_tokens: int) -> tuple[list[dict], int]:
    """Return the messages to send and how many older history messages were left out.

    `history` ends with the new user message. The newest messages that fit the context and
    the proxy's message and image limits are kept, starting at a user turn. Only role and
    content are sent, because the quota proxy rejects any other field.
    """
    if len(history[-1].get("images") or []) > MAX_IMAGES:
        raise ValueError(f"Attach at most {MAX_IMAGES} images per message.")
    system = [{"role": "system", "content": system_prompt.strip()}] if system_prompt.strip() else []
    budget = CONTEXT_TOKENS - max_tokens
    kept: list[dict] = []
    for message in reversed(history):
        candidate = [{"role": message["role"], "content": request_content(message)}, *kept]
        if (
            len(system) + len(candidate) > MAX_MESSAGES
            or image_count(candidate) > MAX_IMAGES
            or token_upper_bound(system + candidate) > budget
        ):
            break
        kept = candidate
    while kept and kept[0]["role"] != "user":
        kept.pop(0)
    if not kept:
        raise ValueError(
            f"This message is too long: with the system prompt, any images, and {max_tokens} output tokens it "
            f"could exceed the {CONTEXT_TOKENS:,}-token context. Shorten it or lower the max output tokens."
        )
    return system + kept, len(history) - len(kept)


def reservation(headers: Mapping[str, str]) -> tuple[int, int] | None:
    """Return the (tokens, micro-USD) reservation that quota-proxy reported, if any."""
    try:
        return int(headers["x-quota-reserved-tokens"]), int(headers["x-quota-reserved-microusd"])
    except (KeyError, ValueError):
        return None


def route(headers: Mapping[str, str]) -> str | None:
    """Return the model that quota-proxy selected, with the reason when it chose for "auto"."""
    model = headers.get("x-routed-model")
    reason = headers.get("x-route-reason")
    if not model:
        return None
    return f"{model} ({reason})" if reason and reason != "requested" else model


def describe_error(error: Exception, host: str) -> str:
    if isinstance(error, openai.APIStatusError):
        detail = _detail(error)
        status = error.status_code
        if status == 401:
            return "The API key was rejected (HTTP 401)."
        if status == 429:
            return f"Caller limit reached (HTTP 429): {detail}. Daily limits reset at 00:00 UTC."
        if status == 400:
            return f"Request rejected (HTTP 400): {detail}"
        if status in (502, 503):
            return f"The inference service is unavailable (HTTP {status}): {detail}"
        return f"Request failed (HTTP {status}): {detail}"
    if isinstance(error, openai.APITimeoutError):
        return f"{host} did not respond in time."
    if isinstance(error, openai.APIConnectionError):
        return (
            f"Could not connect to {host}. A VPN that tunnels Azure address ranges, such as "
            "MSFT-AzVPN-Manual, can block it; see the README."
        )
    return f"The response stream failed: {error}"


def _detail(error: openai.APIStatusError) -> str:
    body = error.body
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
    return error.response.text.strip()[:300] or "no details"
