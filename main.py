from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from PyPDF2 import PdfReader
import pdfplumber
from flask_cors import CORS
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import traceback


app = Flask(__name__)
CORS(app)

# Load API keys
load_dotenv()
os.environ["GOOGLE_API_KEY"]

db = None

# Initialize LLM model
model=GoogleGenerativeAI(model="gemini-2.0-flash")

# Define prompts
prompt1 = ChatPromptTemplate.from_template("""
You are an experienced **Technical Human Resource Manager** specializing in talent acquisition.  
Your task is to **evaluate** the provided resume against the job description.  
### **Instructions**:
- Determine **how well the candidate’s profile aligns** with the job role.  
- Highlight **strengths** (skills, experience, and qualifications that match).  
- Identify **weaknesses** (missing or underdeveloped qualifications).  
- Provide a **concise and professional evaluation** with actionable feedback.
### **Job Description**:
{input}
### **Resume Context**:
{context}
### **Response Format**:
✅ <b>Overall Match Assessment</b>: 
    (Provide a summary of alignment)  
🔹 <b>Key Strengths</b>: 
    (List relevant skills, experience, and achievements)
⚠️ <b>Areas for Improvement</b>: 
    (Mention missing qualifications or weak points)  
📌 <b>Final Verdict</b>:
    (Would you recommend this candidate for the role? Why or why not?)
""")


prompt2 = ChatPromptTemplate.from_template("""
You are an **Applicant Tracking System (ATS) scanner** with expertise in resume parsing and job description matching.  
Your task is to **analyze** the resume against the job description, **identify missing keywords**, and **suggest improvements** to increase the match score.
### **Job Description**:
{input}
### **Resume Content**:
{context}
### **Response Format**:
1️⃣ **Missing Keywords**:  
   - (List missing keywords essential for this role)  
2️⃣ **Why These Keywords Matter**:  
   - (Explain how these missing keywords impact the resume's ATS score and candidate ranking)  
3️⃣ **Recommendations to Improve ATS Score**:  
   - (Actionable steps for adding relevant keywords and enhancing resume content)
""")



prompt3 = ChatPromptTemplate.from_template("""
You are an **Applicant Tracking System (ATS) scanner** that calculates **resume-job fit percentage** based on keyword matching and job relevance.
### **Instructions**:
- **Analyze** the resume against the job description.
- **Calculate a dynamic match percentage** based on skills, experience, and keywords.
- **List missing keywords** that would improve the match.
- **Provide final recommendations** for optimizing the resume.
### **Job Description**:
{input}
### **Resume Context**:
{context}
### **Response Format**:
📊 <b>Match Percentage</b>: 
    (Dynamically calculated value, not static)  
❌ <b>Missing Keywords</b>:  
   - (List of essential keywords missing from the resume)  
📌 <b>Final Recommendations</b>:  
   - (Suggestions for improving the match score)
""")




# Function to process uploaded resume
def extract_all_data(uploaded_file):
    global db

    text = ""

    # Try with pdfplumber first
    try:
        import pdfplumber
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print("pdfplumber failed:", e)

    # If text is still empty, fallback to OCR
    if not text.strip():
        print("Trying OCR fallback...")
        uploaded_file.seek(0)
        images = convert_from_bytes(uploaded_file.read(), dpi=300)
        for image in images:
            text += pytesseract.image_to_string(image)

    if not text.strip():
        raise ValueError("The uploaded PDF contains no extractable text (even after OCR).")

    # Split and embed
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.create_documents([text])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(documents, embedding=embeddings)

    return "Resume successfully processed!"


def get_response(description, db, prompt):  
    document_chain = create_stuff_documents_chain(model, prompt)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    retrieved_docs = retriever.invoke(description)
    response = retrieval_chain.invoke({'input': description, 'context': retrieved_docs})

    return response['answer']


@app.route('/')
def index():
    # return render_template('index.html')
       return jsonify({"message": "Flask API is running on Hugging Face Spaces!"})


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        message = extract_all_data(file)
        return jsonify({"message": message})
    except ValueError as e:
        return jsonify({"error": str(e)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"})



@app.route('/about_resume', methods=['POST'])
def about_resume():
    global db  # Ensure we access the global db variable

    data = request.json
    job_description = data.get('job_description')

    if db is None:  # Check if db exists
        return jsonify({"error": "No resume uploaded yet. Please upload a resume first!"}), 400

    response = get_response(job_description, db, prompt1)
    return jsonify({"response": response})


@app.route('/keywords', methods=['POST'])
def keywords():
    data = request.json
    job_description = data.get('job_description')

    if not db:
        return jsonify({"error": "No resume uploaded yet."})

    response = get_response(job_description, db, prompt2)
    return jsonify({"response": response})


@app.route('/percentage_match', methods=['POST'])
def percentage_match():
    data = request.json
    job_description = data.get('job_description')

    if not db:
        return jsonify({"error": "No resume uploaded yet."})

    response = get_response(job_description, db, prompt3)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)