import streamlit as st
from transformers import pipeline
import io
import docx

def read_docx(file):
    doc = docx.Document(io.BytesIO(file.read()))
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return " ".join(text)

def main():
    st.title("Welcome to CurioQueries")

    page = st.sidebar.selectbox("Select a page", ["Home", "Ask"])

    if page == "Home":
        st.write("This is the main page. Explore and have fun!")

    elif page == "Ask":
        st.header("Ask Your Question")

        file = st.file_uploader('Upload a docx file', type='docx')
        if file is not None:
            context = read_docx(file)

            question = st.text_input('Enter your question')
            if not question.strip(): 
                st.warning("This column can not be empty. Please write your question in the question field.")
            else:
                model_name = "deepset/roberta-base-squad2"
                nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, max_length=50, min_length=30)
                QA_input = {
                    'question': question,
                    'context': context
                }
                res = nlp(QA_input)
                st.write("Answer:", res['answer'])

if __name__ == '__main__':
    main()
