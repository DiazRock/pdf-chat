import os
import streamlit as st
from pypdf import PdfReader
import numpy as np
from faiss import IndexFlatL2
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import time


@st.cache_resource
def get_client():
    api_key = os.environ["MISTRAL_API_KEY"]
    return MistralClient(api_key=api_key)


CLIENT: MistralClient = get_client()


@st.cache_data
def embed(text: str):
    return (
        CLIENT.embeddings("mistral-embed", text)
        .data[0]
        .embedding
    )


def add_message(msg, agent="ai", stream=True, store=True):
    if stream and isinstance(msg, str):
        msg = stream_str(msg)

    with st.chat_message(agent):
        if stream:
            output = st.write_stream(msg)
        else:
            output = msg
            st.write(msg)
    
    if store:
        st.session_state.messages.append(
            dict(agent=agent, content=output)
        )


def stream_response(response):
    for r in response:
        yield r.choices[0].delta.content


def stream_str(s, speed=250):
    for c in s:
        yield c
        time.sleep(1 / speed)


PROMPT = """
An excerpt from a document is given below

----------------------
{context}
----------------------

Given the document excerpt, answer the following query.
If the context does not provide enough information, decline to answer.
Do not output anything that can't be answered from the context.

Query: {query}
Answer:
"""

def build_index():
    st.session_state.messages = []

    pdf_file = st.session_state.pdf_file

    if not pdf_file:
        st.session_state.clear()
        return
    
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    
    st.session_state.text = text

    st.sidebar.info(
        f"The uploaded PDF has {len(reader.pages)} pages"
        f"and {len(text)} characters"
    )
    chunk_size = 1024
    chunks = [
        text[i: i + 2 * chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

    if len(chunks) > 100:
        st.error("Document is too long!")
        st.session_state.clear()
        return

    st.sidebar.info(f"Indexing {len(chunks)} chunks.")
    progress = st.sidebar.progress(0)

    embeddings = []
    for i, chunk in enumerate(chunks):
        embeddings.append(embed(chunk))
        progress.progress((i + 1) / len(chunks))

    embeddings = np.array(embeddings)
    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(embeddings)

    st.session_state.index = index
    st.session_state.chunks = chunks


def reply(query: str, index: IndexFlatL2):
    embedding = embed(query)
    embedding = np.array([embedding])

    _, indexes = index.search(embedding, k = len(embedding))
    context = [
        st.session_state.chunks[i] for i in indexes.tolist()[0]
    ]

    messages = [
        ChatMessage(
            role="user",
            content = PROMPT.format(context = context, query= query),
        )
    ]
    response = CLIENT.chat_stream(
        model = "mistral-small", messages=messages
    )

    add_message(stream_response(response))
