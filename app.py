import streamlit as st
from transformers import pipeline
import os
import io
import docx

def read_docx(file):
    doc = docx.Document(io.BytesIO(file.read()))
    text = [paragraph.text for paragraph in doc.paragraphs]
    return " ".join(text)

def load_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.docx')]
    return files

def chunk_text(text, max_length=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks

def main():
    st.title("Welcome to CurioQueries for Engineers")

    page = st.sidebar.selectbox("Select a page", ["Home", "Ask"])

    if page == "Home":
        st.write("This is the main page. Explore and have fun!")

    elif page == "Ask":
        st.header("Ask Your Question")

        directory = st.text_input("Enter the directory containing the .docx files")
        question = st.text_input('Enter your question')

        if not directory and not question.strip():
            st.warning("Please enter a directory and write your question in the question field.")
        elif not directory:
            st.warning("Please enter a directory.")
        elif not question.strip():
            st.warning("This column cannot be empty. Please write your question in the question field.")
        else:
            if not os.path.exists(directory):
                st.error("The specified directory does not exist.")
            else:
                files = load_files(directory)
                if not files:
                    st.warning("No .docx files found in the specified directory.")
                else:
                    model_name = "deepset/roberta-base-squad2"
                    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

                    answers = []
                    for file_name in files:
                        file_path = os.path.join(directory, file_name)
                        with open(file_path, 'rb') as file:
                            context = read_docx(file)
                            chunks = chunk_text(context)

                            for chunk in chunks:
                                QA_input = {'question': question, 'context': chunk}
                                res = nlp(QA_input)
                                score = res['score']
                                answers.append((file_name, res['answer'], score))

                    # Sort answers by score in descending order and take top 3
                    top_answers = sorted(answers, key=lambda x: x[2], reverse=True)[:3]

                    st.write("Top Answers found:")
                    for file_name, answer, score in top_answers:
                        st.write(f"**{file_name}**: {answer} (Score: {score})")

if __name__ == '__main__':
    main()

