import json
import traceback

import pandas as pd
import streamlit as st

from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging
from src.mcqgenerator.utils import get_table_data, read_file

#Loading json file
with open("Response.json", "r") as f:
    RESPONSE_JSON = json.load(f)

#Creating title for the app
st.title("MCQs Creator Application with Langchain")

#Create a form using st.form
with st.form("user_inputs"):
    #File upload
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    #Input fields
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50, value=5)

    #Subject
    subject = st.text_input("Subject", max_chars=30)

    #Quiz Tone
    tone = st.text_input("Complexity level of questions", max_chars=20, placeholder="Simple")

    #Add button
    button = st.form_submit_button("Create MCQs")

if button:
    if uploaded_file is None or not subject or not tone:
        st.warning("Please upload a file and fill in the subject and complexity level.")
    else:
        with st.spinner("Generating MCQs..."):
            try:
                text = read_file(uploaded_file)

                result = generate_evaluate_chain(
                    {
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON),
                    }
                )
            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                logging.error(f"Streamlit app error: {e}")
                st.error(f"Something went wrong: {e}")
            else:
                quiz = result.get("quiz")
                table_data = get_table_data(quiz) if quiz else None

                if table_data:
                    df = pd.DataFrame(table_data)
                    df.index = df.index + 1
                    st.table(df)
                    st.text_area(label="Review", value=result.get("review", ""), height=150)
                else:
                    st.error("Could not read the quiz from the model's response.")
                    st.text_area(label="Raw response", value=str(quiz), height=200)
