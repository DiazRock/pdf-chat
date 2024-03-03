import streamlit as st
from pypdf import PdfReader

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
