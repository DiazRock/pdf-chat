import streamlit as st
from build_index import build_index, add_message, reply
from faiss import IndexFlatL2


st.title("ChatGPT-like clone")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["agent"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)


if st.sidebar.button("Reset conversasion"):
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["agent"]):
        st.write(message["content"])

if not "text" in st.session_state:
    add_message(
        """
            This is a simple demonstration of how to use large language model and a vector database
            to implement a bare-bones chat-with-your-pdf application

        """,
        store=False
    )

    add_message(
        "To begin, please upload your PDF file in the sidebar",
        store=False
    )


pdf = st.sidebar.file_uploader(
    "Upload a PDF file",
    type="PDF",
    key="pdf_file",
    on_change=build_index
)

if not pdf:
    st.stop()


index: IndexFlatL2 = st.session_state.index
query = st.chat_input("Ask something about your PDF")


if not st.session_state.messages:
    reply(
        "In one sentence, what is this document about?",
        index,
    )
    add_message("Ready to answer your questions.")


if query:
    add_message(query, agent="human", stream=False, store=True)
    reply(query, index)
