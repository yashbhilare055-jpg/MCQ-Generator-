import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.mcqgenerator.logger import logging

load_dotenv()
KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=KEY,
    temperature=0.5
)

QUIZ_TEMPLATE = """
Text: {text}

You are an expert MCQ Maker. Given the above text. Create a Quiz of {number} multiple choice questions for {subject} students in a {tone} tone.
Make sure the questions are not repeated an check that all the questions conform to the text.
Format your response exactly like RESPONSE_JSON below and use it as a guide.
Return only the JSON, with {number} MCQs.

###RESPONSE_JSON
{response_json}
"""

quiz_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=QUIZ_TEMPLATE,
)

quiz_chain = quiz_prompt | llm | StrOutputParser()

REVIEW_TEMPLATE = """

You are an expert English grammarian and writer. Given a multiple choice quiz for {subject} students,
evaluate the complexity of the questions and give a complete analysis of the quiz in at most 50 words.
Only use the maximum 50 words for the complexity analysis.
If the quiz is not at par with the cognitive and analytical abilities of the students,
update the quiz questions that need to be changed and adjust the tone so that it perfectly fits the students' abilities.

Quiz_MCQs:
{quiz}

Check from an expert English writer of the above quiz:

"""

review_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=REVIEW_TEMPLATE,
)

review_chain = review_prompt | llm | StrOutputParser()

def generate_evaluate_chain(inputs):
    """
    inputs = {
        "text": ..., "number": ..., "subject": ..., "tone": ..., "response_json": ...
    }
    Returns {"quiz": <json string>, "review": <text>}
    """

    try:
        logging.info("Generating quiz...")
        quiz = quiz_chain.invoke(inputs)

        logging.info("Reviewing quiz...")
        review = review_chain.invoke({"subject": inputs["subjects"], "quiz": quiz})

        logging.info("Quiz generation and review completed")
        return {"quiz": quiz, "review": review}
    
    except Exception as e:
        logging.error(f"MCQ Generator failed: {e}")
        raise