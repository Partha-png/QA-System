# app.py

import streamlit as st
from llmengine import process_txt, ask_question
import os

st.set_page_config(page_title="Gemini PDF QA", page_icon="📄")
st.title("📄 Ask Questions from your txt (Gemini LLM)")

uploaded_file = st.file_uploader("Upload a PDF", type=['txt'])
question = st.text_input("Ask your question:")

if uploaded_file is not None and question:
    with st.spinner("Processing PDF and answering your question..."):
        # Save uploaded file to temp
        temp_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # LLM processing
        index = process_txt(temp_path)
        answer = ask_question(index, question)

        st.markdown("### 📢 Answer:")
        st.write(answer)
