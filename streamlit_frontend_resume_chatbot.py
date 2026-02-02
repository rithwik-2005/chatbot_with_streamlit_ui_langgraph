import streamlit as st
import uuid

from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# Utility functions
# ============================================================

def generate_thread_id():
    """Generate a unique conversation ID"""
    return str(uuid.uuid4())


def add_thread(thread_id):
    """Add a thread ID to sidebar list if not already present"""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    """Start a new conversation"""
    new_thread_id = generate_thread_id()
    st.session_state["thread_id"] = new_thread_id
    add_thread(new_thread_id)
    st.session_state["message_history"] = []


def load_conversation(thread_id):
    """Load messages from LangGraph memory"""
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


# ============================================================
# Session State Initialization (ORDER MATTERS!)
# ============================================================

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    add_thread(st.session_state["thread_id"])


# ============================================================
# Sidebar UI
# ============================================================

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("NEW CHAT"):
    reset_chat()

st.sidebar.header("My Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(thread_id):
        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)
        ui_messages = []

        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            ui_messages.append(
                {"role": role, "content": msg.content}
            )

        st.session_state["message_history"] = ui_messages


# ============================================================
# Main Chat UI
# ============================================================

# Replay chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Save user message
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # LangGraph config
    config = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    # Convert FULL UI history to LangChain messages
    langchain_messages = []
    for msg in st.session_state["message_history"]:
        if msg["role"] == "user":
            langchain_messages.append(
                HumanMessage(content=msg["content"])
            )
        else:
            langchain_messages.append(
                AIMessage(content=msg["content"])
            )

    # Stream assistant response
    with st.chat_message("assistant"):
        ai_chunks = []

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": langchain_messages},
                config=config,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    ai_chunks.append(message_chunk.content)
                    yield message_chunk.content

        st.write_stream(ai_only_stream())

    # Save assistant message
    final_ai_message = "".join(ai_chunks)
    st.session_state["message_history"].append(
        {"role": "assistant", "content": final_ai_message}
    )
