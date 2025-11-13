import os
import json
import re
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from duckduckgo_search import DDGS

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA

import pathlib

# Load environment variables
load_dotenv()

# Load CSS
def load_css(file_path):
    with open(file_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("report.css")
load_css(css_path)


st.title("🩺Understand Your Medical Report")

pdf_file = st.file_uploader("📄 Upload your Medical Report (PDF)", type=["pdf"], key="report_uploader")
generate_button = st.button("🔍 Generate Explanation")

# PDF Text Extraction
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    return "".join([page.extract_text() or "" for page in reader.pages])

# Image search (return 3 images)
def search_multiple_medical_images(diagnosis, explanation):
    focused_prompt = f"{diagnosis} medical diagram"
    try:
        with DDGS() as ddgs:
            time.sleep(5)
            results = ddgs.images(focused_prompt, max_results=3)
            return [result["image"] for result in results]
    except Exception as e:
        print("Image search failed:", e)
    return []

# Embedding model
embedding_model = AzureOpenAIEmbeddings(
    openai_api_key=os.getenv("Embeddings_AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("Embeddings_AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("Embeddings_AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("Embeddings_AZURE_OPENAI_API_VERSION"),
    chunk_size=1000
)

# Main logic
if pdf_file and generate_button:
    with st.spinner("⚙️ Extracting and interpreting your report..."):
        raw_text = extract_text_from_pdf(pdf_file)
        if not raw_text.strip():
            st.error("No text found in the uploaded PDF.")
        else:
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_text(raw_text)
            documents = [Document(page_content=chunk) for chunk in chunks]
            vector_db = FAISS.from_documents(documents, embedding_model)
            retriever = vector_db.as_retriever()

            llm = AzureChatOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                temperature=0.7,
                max_tokens=1500,
            )

            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            user_prompt = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the medical image and structure your response as follows:

### 1. Image Type & Region
- Identify imaging modality (X-ray/MRI/CT/Ultrasound/etc.).
- Specify anatomical region and positioning.
- Evaluate image quality and technical adequacy.

### 2. Key Findings
- Highlight primary observations systematically.
- Identify potential abnormalities with detailed descriptions.
- Include measurements and densities where relevant.

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level.
- List differential diagnoses ranked by likelihood.
- Support each diagnosis with observed evidence.
- Highlight critical/urgent findings.

### 4. Patient-Friendly Explanation
- Simplify findings in clear, non-technical language.
- Avoid medical jargon or provide easy definitions.
- Include relatable visual analogies.

### 5. Research Context
- Use DuckDuckGo search to find recent medical literature.
- Search for standard treatment protocols.
- Provide 2-3 key references supporting the analysis.

Ensure a structured and medically accurate response using clear markdown formatting.
"""

            try:
                response = qa_chain.invoke(user_prompt)
                result_text = response["result"]
                json_match = re.search(r'{.*}', result_text, re.DOTALL)

                if json_match:
                    diagnosis_block = json.loads(json_match.group())
                    diagnosis = diagnosis_block.get("diagnosis", "").strip()
                    explanation = result_text.replace(json_match.group(), "").strip()
                else:
                    diagnosis = ""
                    explanation = result_text.strip()
            except Exception:
                st.warning("⚠️ Could not extract structured data. Showing full response.")
                diagnosis = ""
                explanation = response.get("result", "")

            # Save to session for later use
            st.session_state["diagnosis"] = diagnosis
            st.session_state["explanation"] = explanation
            st.session_state["image_urls"] = search_multiple_medical_images(diagnosis, explanation)

            # Display only explanation here
            st.subheader("Layman Explanation")
            if diagnosis:
                st.markdown(f"**🧠 Diagnosis:** `{diagnosis}`")
            st.markdown(explanation)
