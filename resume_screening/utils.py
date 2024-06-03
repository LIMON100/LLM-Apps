# utils - By: limon - Sun Jun 2 2024

import openai
from langchain.schema import Document
from pypdf import PdfReader
import uuid
import google.generativeai as genai
from IPython.display import Markdown
import os
import textwrap
import uuid
import re
from sklearn.metrics.pairwise import cosine_similarity

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def init_model():
    os.environ['GOOGLE_API_KEY'] = "AIzaSyCHghstLR5_5X8p7FN5Jx1VMzXKi_CmVGU"
    genai.configure(api_key = os.environ['GOOGLE_API_KEY'])

    # Initialize Gemini model
    model = genai.GenerativeModel('gemini-1.5-flash')
    return model


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def make_docs(user_pdf_list, unique_id):
    docs = []
    for filename in user_pdf_list:
        chunks = get_pdf_text(filename)
        docs.append(Document(
            page_content=chunks,
            metadata={"name": filename.name, "id": str(uuid.uuid4()), "type": filename.type, "size": filename.size, "unique_id": unique_id},
        ))
    return docs


def generate_summary(job_description, doc, model):
    prompt = "I am looking for a good deep learning engineer, which have saveral years of experience, building many models, deploy scalable model in real world."
    evaluation_prompt = f"Please evaluate the following answer to the prompt:\n\nPrompt: {job_description}\n\nAnswer: {doc.page_content}\n\nMake a summary of the response and try to match with the prompt and give a score that can judge the response.You might use similarity score or something which can help better to judge the response with prompt. Only return the score, what is the main expertise of the candidate and overall summary"

    response = model.generate_content(evaluation_prompt)
    return response.text


# Function to embed content
def embed_content(content):
    return genai.embed_content(
        model="models/embedding-001",
        content=content,
        task_type="retrieval_document",
        title="Embedding of single string"
    )


def extract_relevant_info(summary):
    score = re.search(r'Score: (\d+/10)', summary)
    main_expertise = re.search(r'Main Expertise: ([^\n]+)', summary)
    overall_summary = re.search(r'Overall Summary: ([^\n]+)', summary)
    detailed_summary = re.search(r'Overall Summary: [^\n]+\n([^\n]+)', summary)

    return {
        'score': score.group(1) if score else "N/A",
        'main_expertise': main_expertise.group(1) if main_expertise else "N/A",
        'summary': detailed_summary.group(1) if detailed_summary else "N/A",
        'overall_summary': overall_summary.group(1) if overall_summary else "N/A"
    }


def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]