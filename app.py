import streamlit as st
from transformers import pipeline
import os
import io
import docx2txt

def read_docx(file):
    try:
        doc = docx.Document(io.BytesIO(file.read()))
        text = [paragraph.text for paragraph in doc.paragraphs]
        return " ".join(text)
    except Exception as e:
        st.error(f"Error reading docx file: {e}")
        return ""

def load_files(directory):
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.docx')]
        return files
    except Exception as e:
        st.error(f"Error loading files from directory: {e}")
        return []

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
        st.header("Home")
        st.write("This is the home page.")
    elif page == "Ask":
        st.header("Ask Your Question")

        file = st.file_uploader('Upload a docx file', type='docx')
        directory = st.text_input("Enter the directory containing the .docx files")
        question = st.text_input('Enter your question')

        if not file and not directory:
            st.warning("Please upload a docx file or enter a directory.")
        elif not question.strip():
            st.warning("Please write your question in the question field.")
        else:
            if file:
                context = read_docx(file)
                if context:
                    try:
                        model_name = "deepset/roberta-base-squad2"
                        nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
                        QA_input = {'question': question, 'context': context}
                        res = nlp(QA_input)
                        st.write("Answer:", res['answer'])
                    except Exception as e:
                        st.error(f"Error processing question-answering pipeline: {e}")
            
            if directory:
                if not os.path.exists(directory):
                    st.error("The specified directory does not exist.")
                else:
                    files = load_files(directory)
                    if not files:
                        st.warning("No .docx files found in the specified directory.")
                    else:
                        try:
                            model_name = "deepset/roberta-base-squad2"
                            nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

                            answers = []
                            for file_name in files:
                                file_path = os.path.join(directory, file_name)
                                with open(file_path, 'rb') as file:
                                    context = read_docx(file)
                                    if context:
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
                        except Exception as e:
                            st.error(f"Error processing files in directory: {e}")

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()

