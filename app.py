import streamlit as st

from main import (
    load_documents,
    split_documents,
    create_vector_store,
    create_rag_components,
    ask_question,
    validate_config,
    PDF_PATH
)


# PAGE CONFIG

st.set_page_config(
    page_title="Python Tutor",
    layout="wide"
)


# STARTUP VALIDATION

try:
    validate_config()
except (RuntimeError, FileNotFoundError) as e:
    st.error(f"⚠️ Setup problem: {e}")
    st.stop()


# SESSION STATE

if "chats" not in st.session_state:
    st.session_state.chats = []

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None


# LOAD RAG ONCE

@st.cache_resource
def initialize_rag():

    documents = load_documents(PDF_PATH)

    chunks = split_documents(documents)

    vector_store = create_vector_store(chunks)

    llm, prompt = create_rag_components()

    return vector_store, llm, prompt


with st.spinner("Loading Python Tutor..."):

    try:
        vector_store, llm, prompt = initialize_rag()
    except Exception as e:
        st.error(f"⚠️ Failed to initialize the tutor: {e}")
        st.stop()


# HEADER

st.title("Python Tutor")


# SIDEBAR

st.sidebar.title("Python Tutor")


# NEW CHAT

if st.sidebar.button("➕ New Chat"):

    new_chat = {
        "title": "New Chat",
        "messages": []
    }

    st.session_state.chats.insert(
        0,
        new_chat
    )

    st.session_state.current_chat = 0

    st.rerun()

st.sidebar.markdown("---")

st.sidebar.title("💬 Chat History")


# CHAT LIST

if len(st.session_state.chats) == 0:

    st.sidebar.info(
        "No chat history yet."
    )

else:

    for idx, chat in enumerate(
        st.session_state.chats
    ):

        title = chat["title"]

        if st.sidebar.button(
            title,
            key=f"chat_{idx}"
        ):
            st.session_state.current_chat = idx
            st.rerun()


# CLEAR CURRENT CHAT

st.sidebar.markdown("---")

if (
    st.session_state.current_chat
    is not None
):

    if st.sidebar.button(
        "🗑️ Delete Current Chat"
    ):

        st.session_state.chats.pop(
            st.session_state.current_chat
        )

        if len(st.session_state.chats) == 0:
            st.session_state.current_chat = None
        else:
            st.session_state.current_chat = 0

        st.rerun()


# DISPLAY CURRENT CHAT

if (
    st.session_state.current_chat
    is not None
):

    current_chat = (
        st.session_state.chats[
            st.session_state.current_chat
        ]
    )

    for message in current_chat["messages"]:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


# CHAT INPUT

user_question = st.chat_input(
    "Ask a Python question..."
)


# HANDLE QUESTION

if user_question:

    # Create first chat automatically

    if (
        st.session_state.current_chat
        is None
    ):

        new_chat = {
            "title": user_question[:40],
            "messages": []
        }

        st.session_state.chats.insert(
            0,
            new_chat
        )

        st.session_state.current_chat = 0

    current_chat = (
        st.session_state.chats[
            st.session_state.current_chat
        ]
    )

    # Set title from first question

    if (
        current_chat["title"]
        == "New Chat"
    ):

        if len(user_question) > 40:

            current_chat["title"] = (
                user_question[:40] + "..."
            )

        else:

            current_chat["title"] = (
                user_question
            )

    # Save User Message

    current_chat["messages"].append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # Show User Message

    with st.chat_message("user"):

        st.markdown(
            user_question
        )

    # Generate Response

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Thinking..."
        ):

            answer = ask_question(
                llm=llm,
                prompt=prompt,
                vector_store=vector_store,
                question=user_question,
                student_level="Beginner",
                weak_area="General Python"
            )

            st.markdown(
                answer
            )

    # Save Response

    current_chat["messages"].append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    st.rerun()
