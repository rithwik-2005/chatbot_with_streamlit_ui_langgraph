# ===============================================================
# STREAMLIT + LANGGRAPH CHAT APPLICATION
# ===============================================================
# IMPORTANT CONCEPTS TO REMEMBER:
#
# 1. Streamlit reruns this script from top to bottom
#    on EVERY user interaction (typing, clicking, etc.).
#
# 2. st.session_state does NOT clear during reruns.
#    It ONLY clears when:
#       - Browser tab is refreshed (F5)
#       - Browser tab is closed
#       - Streamlit server restarts
#
# 3. One UUID = One conversation (NOT one UUID per message)
#
# 4. message_history is ONLY for UI display.
#    LangGraph memory is DIFFERENT and is tied to thread_id.
#
# 5. LLMs are STATELESS.
#    That is why we send the FULL conversation each time.
#
# ===============================================================

import streamlit as st
#we imported uuid to generate unique conversation IDs
import uuid 
#import chatbot object from langgraph_backend
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage,AIMessage

# ===============================================================
# 1. CREATE OR REUSE A UUID (CONVERSATION ID)
# ===============================================================
# Streamlit reruns the script many times.
# If we generate a new UUID every time → LangGraph memory breaks.
#
# So:
# - Create UUID ONLY ONCE
# - Store it in st.session_state
# - Reuse it on every rerun
#
# This UUID identifies ONE conversation.
# ===============================================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id=str(uuid.uuid4())

# Configuration object passed to LangGraph
# LangGraph uses thread_id to:
# - Load previous graph state
# - Continue the same conversation
CONFIG = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}
# ===============================================================
# 2. INITIALIZE UI CHAT HISTORY
# ===============================================================
# message_history:
# - Exists ONLY for UI rendering
# - Stores what the user sees on screen
#
# This is NOT LangGraph memory.
#
# Example:
# [
#   {"role": "user", "content": "Hi"},
#   {"role": "assistant", "content": "Hello"}
# ]
# ===============================================================
if "message_history" not in st.session_state:
    st.session_state.message_history = []

# ===============================================================
# 3. CONVERT UI MESSAGES → LANGCHAIN MESSAGES
# ===============================================================
# Streamlit stores messages as dictionaries.
# LangGraph requires LangChain message objects.
#
# This function converts UI history into the format
# that LangGraph + LLM understand.
# ===============================================================
def to_langchain_messages(history):
    messages = []

    for msg in history:
        # User message → HumanMessage
        if msg["role"] == "user":
            messages.append(
                HumanMessage(content=msg["content"])
            )
        # Assistant message → AIMessage
        else:
            messages.append(
                AIMessage(content=msg["content"])
            )

    return messages

# ===============================================================
# 4. RE-RENDER CHAT UI ON EVERY RERUN
# ===============================================================
# Streamlit rebuilds the UI every rerun.
# This loop redraws the entire conversation
# using message_history stored in session_state.
# ===============================================================
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ===============================================================
# 5. CHAT INPUT BOX
# ===============================================================
# st.chat_input():
# - Shows an input field
# - Returns None until user submits a message
# ===============================================================
user_input = st.chat_input("Type your message here...")
# ===============================================================
# 6. HANDLE USER MESSAGE
# ===============================================================
if user_input:
    # -----------------------------------------------------------
    # Save user message in UI history
    # -----------------------------------------------------------
    st.session_state.message_history.append(
        {"role": "user", "content": user_input}
    )

    # -----------------------------------------------------------
    # Display user message immediately
    # -----------------------------------------------------------
    with st.chat_message("user"):
        st.markdown(user_input)

    # -----------------------------------------------------------
    # Convert FULL UI history to LangChain format
    #
    # Why full history?
    # Because LLMs are stateless and forget everything
    # unless we resend the conversation.
    # -----------------------------------------------------------
    messages = to_langchain_messages(
        st.session_state.message_history
    )

    # -----------------------------------------------------------
    # Call LangGraph
    #
    # What LangGraph does internally:
    # 1. Uses thread_id to load memory
    # 2. Applies graph logic (conditions, tools, flow)
    # 3. Calls the LLM with proper context
    # -----------------------------------------------------------
    response = chatbot.invoke(
        {"messages": messages},
        config=CONFIG
    )

    # -----------------------------------------------------------
    # Extract the AI's latest response
    # -----------------------------------------------------------
    ai_message = response["messages"][-1].content

    # -----------------------------------------------------------
    # Save AI message in UI history
    # -----------------------------------------------------------
    st.session_state.message_history.append(
        {"role": "assistant", "content": ai_message}
    )

    # -----------------------------------------------------------
    # Display AI message
    # -----------------------------------------------------------
    with st.chat_message("assistant"):
        st.markdown(ai_message)