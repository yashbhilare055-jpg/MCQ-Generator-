import json
import traceback

import PyPDF2

from src.mcqgenerator.logger import logging


def read_file(file):
    """Read text from an uploaded PDF or TXT file (Streamlit upload object)."""
    if file.name.endswith(".pdf"):
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            logging.error(f"Error reading PDF: {e}")
            raise Exception("Error reading the PDF file")

    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")

    else:
        raise Exception("Unsupported file format. Only PDF and TXT files are supported.")


def get_table_data(quiz_str):
    """Convert the quiz JSON string from the LLM into a list of rows for a DataFrame."""
    try:
        # Gemini often wraps JSON in ```json ... ``` fences, so strip them
        cleaned = quiz_str.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        quiz_dict = json.loads(cleaned)
        quiz_table_data = []

        for _, value in quiz_dict.items():
            mcq = value["mcq"]
            options = " | ".join(
                f"{opt}: {opt_value}" for opt, opt_value in value["options"].items()
            )
            correct = value["correct"]
            quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})

        return quiz_table_data

    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        logging.error(f"Error parsing quiz data: {e}")
        return False
