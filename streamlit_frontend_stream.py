import streamlit as st
import uuid

from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage


# ===============================================================
# 1. UUID FOR ONE CONVERSATION
# ===============================================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

CONFIG = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# ===============================================================
# 2. UI CHAT HISTORY (DISPLAY ONLY)
# ===============================================================
if "message_history" not in st.session_state:
    st.session_state.message_history = []


# ===============================================================
# 3. CONVERT UI HISTORY → LANGCHAIN FORMAT
# ===============================================================
def to_langchain_messages(history):
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages


# ===============================================================
# 4. REPLAY CHAT UI
# ===============================================================
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ===============================================================
# 5. USER INPUT
# ===============================================================
user_input = st.chat_input("Type your message here...")


# ===============================================================
# 6. HANDLE USER MESSAGE + STREAM RESPONSE
# ===============================================================
if user_input:
    # Save user message
    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Convert full conversation
    messages = to_langchain_messages(
        st.session_state.message_history
    )

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()        # UI container
        full_ai_response = ""           # Accumulate tokens

        for message_chunk, metadata in chatbot.stream(
            {"messages": messages},
            config=CONFIG,
            stream_mode="messages"
        ):
            if isinstance(message_chunk, AIMessage):
                full_ai_response += message_chunk.content
                placeholder.markdown(full_ai_response)

    # Save final assistant message
    st.session_state.message_history.append(
        {"role": "assistant", "content": full_ai_response}
    )
