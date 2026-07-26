import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import json

# دالة استخراج النص من PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# واجهة Streamlit
st.title("CV Parser Project")

uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")

if uploaded_file is not None:
    cv_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text")
    st.text(cv_text[:500])  # عرض أول 500 حرف

    # نموذج من Hugging Face
    nlp = pipeline("text-generation", model="google/flan-t5-large")

    prompt = f"""
    Extract the following details from the CV text and return them in JSON format:
    - full_name
    - email
    - education (list of {{degree, institution, year}})
    - skills (list of strings)
    - experience (list of {{role, company, years}})

    CV Text:
    {cv_text}
    """

    response = nlp(prompt, max_length=512)[0]['generated_text']

    try:
        parsed_output = json.loads(response)
    except:
        parsed_output = response

    st.subheader("Parsed JSON Output")
    st.json(parsed_output)
