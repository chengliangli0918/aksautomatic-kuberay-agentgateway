"""Streamlit chat client for the quota-limited vLLM API behind agentgateway.

Run it from this directory so .streamlit/config.toml keeps it on localhost:
    streamlit run app.py
"""

from __future__ import annotations

import os
import pathlib
import urllib.parse

import openai
import streamlit as st

import chat

st.set_page_config(page_title="AgentGateway POC", page_icon="💬")
# Indent replies a little and use a smaller font so they stand apart from the questions.
REPLY = '[data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"])'
st.html(
    f"<style>{REPLY} {{ margin-left: 2rem; width: calc(100% - 2rem); }}"
    f' {REPLY} [data-testid="stMarkdownContainer"] {{ font-size: 0.875rem; }}'
    f' {REPLY} [data-testid="stCode"] code, {REPLY} [data-testid="stCaptionContainer"] {{ font-size: 0.75rem; }}'
    "</style>"
)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("totals", {"tokens": 0, "microusd": 0})


def make_client(base_url: str, api_key: str) -> openai.OpenAI:
    # No automatic retries: every admitted chat request reserves quota, so a retry would
    # charge the caller again.
    return openai.OpenAI(
        base_url=base_url, api_key=api_key, max_retries=0, timeout=openai.Timeout(600.0, connect=10.0)
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_models(base_url: str, api_key: str) -> list[str]:
    # Model listing is authenticated but does not reserve quota.
    return [model.id for model in make_client(base_url, api_key).models.list()]


def environment_api_key() -> str:
    key = os.environ.get("INFERENCE_API_KEY", "").strip()
    path = os.environ.get("INFERENCE_API_KEY_FILE", "").strip()
    if not key and path:
        try:
            key = pathlib.Path(path).expanduser().read_text(encoding="utf-8").strip()
        except OSError as error:
            st.error(f"Cannot read INFERENCE_API_KEY_FILE: {error.strerror}")
    return key


def stream_reply(client: openai.OpenAI, request: dict, state: dict):
    with client.chat.completions.create(**request, stream=True) as stream:
        state["reservation"] = chat.reservation(stream.response.headers)
        for chunk in stream:
            text = chunk.choices[0].delta.content if chunk.choices else None
            if text:
                state["text"] += text
                yield text


def reservation_text(tokens: int, microusd: int) -> str:
    return f"Reserved {tokens:,} tokens · ${microusd / 1_000_000:,.4f} chargeback"


with st.sidebar:
    st.header("Connection")
    endpoint = st.text_input("Endpoint", value=os.environ.get("INFERENCE_URL", ""), placeholder="https://<hostname>")
    default_key = environment_api_key()
    # A key from the environment stays on the server; it is never sent to the browser.
    typed_key = st.text_input(
        "API key",
        type="password",
        placeholder="Using the key from the environment" if default_key else "Paste a caller API key",
    )
    api_key = typed_key.strip() or default_key
    host = urllib.parse.urlsplit(endpoint.strip()).hostname or endpoint.strip()
    base_url, models = None, []
    if endpoint.strip():
        try:
            base_url = chat.api_base_url(endpoint)
        except ValueError as error:
            st.error(str(error))
    if base_url and not api_key:
        st.info("Enter an API key.")
    elif base_url:
        try:
            models = fetch_models(base_url, api_key)
        except openai.APIError as error:
            st.error(chat.describe_error(error, host))
    model = st.selectbox("Model", models) if models else None

    st.header("Generation")
    system_prompt = st.text_area("System prompt", help="Optional. It counts against the context window.")
    max_tokens = st.slider("Max output tokens", 1, chat.MAX_OUTPUT_TOKENS, 256)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)

    st.header("Quota")
    st.caption(
        f"Each request that the gateway accepts reserves {chat.CONTEXT_TOKENS:,} tokens of the "
        "caller's daily allowance, even if it fails later. Daily limits reset at 00:00 UTC."
    )
    quota_box = st.empty()

if not chat.is_loopback(st.get_option("server.address")):
    st.warning(
        "Other machines may be able to open this app and spend your quota. "
        "Run it from the client directory so .streamlit/config.toml binds it to 127.0.0.1."
    )

st.title("AgentGateway POC")
st.caption(f"{model} · {host}" if model else "Set the endpoint and API key in the sidebar.")
clear_slot = st.empty()


def clear_history() -> None:
    st.session_state.messages = []


def show_turn(message: dict) -> None:
    if message.get("images"):
        st.image(message["images"], width=240)
    if message["content"].strip():
        st.markdown(message["content"])


NOT_ANSWERED = "Not answered; left out of later requests."

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        show_turn(message)
        if message.get("error"):
            st.error(message["error"])
        if message.get("reservation"):
            st.caption(reservation_text(*message["reservation"]))
        if message.get("excluded") and message["role"] == "assistant":
            st.caption(NOT_ANSWERED)

prompt = st.chat_input(
    "Message the model, or paste or attach images",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp", "gif", "bmp"],
    max_upload_size=20,
    disabled=model is None,
)
if prompt and (prompt.text.strip() or prompt.files):
    try:
        images = [chat.image_data_url(file.getvalue()) for file in prompt.files]
        turn = {"role": "user", "content": prompt.text, "images": images}
        history = [m for m in st.session_state.messages if not m.get("excluded")]
        messages, dropped = chat.request_messages(system_prompt, [*history, turn], max_tokens)
    except ValueError as error:
        st.error(str(error))
    else:
        st.session_state.messages.append(turn)
        with st.chat_message("user"):
            show_turn(turn)
        if dropped:
            st.info(
                f"Left out the {dropped} oldest messages so the request fits the "
                f"{chat.CONTEXT_TOKENS:,}-token context and the {chat.MAX_IMAGES}-image limit."
            )
        state = {"text": "", "reservation": None, "error": None}
        request = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        with st.chat_message("assistant"):
            try:
                st.write_stream(stream_reply(make_client(base_url, api_key), request, state))
            except openai.APIError as error:
                if isinstance(error, openai.APIStatusError):
                    # Requests that fail after admission still report their reservation.
                    state["reservation"] = state["reservation"] or chat.reservation(error.response.headers)
                state["error"] = chat.describe_error(error, host)
                st.error(state["error"])
            if state["reservation"]:
                tokens, microusd = state["reservation"]
                st.session_state.totals["tokens"] += tokens
                st.session_state.totals["microusd"] += microusd
                st.caption(reservation_text(tokens, microusd))
            if not state["text"]:
                turn["excluded"] = True
                st.caption(NOT_ANSWERED)
        # A failed request stays in the conversation with its error, but is never sent again.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": state["text"],
                "reservation": state["reservation"],
                "error": state["error"],
                "excluded": not state["text"],
            }
        )

totals = st.session_state.totals
quota_box.markdown(
    f"**Reserved this session:** {totals['tokens']:,} tokens · ${totals['microusd'] / 1_000_000:,.4f}"
)
if st.session_state.messages:
    # Drawn last, at the top of the page, so it also appears right after the first reply. The
    # callback clears the history before the rerun, so the button disappears with it.
    clear_slot.button("Clear chat history", key="clear_history", icon=":material/delete:", on_click=clear_history)
