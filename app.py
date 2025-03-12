import streamlit as st
import fitz  # PyMuPDF for PDF extraction
import os, time, difflib, base64, string, re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
from fpdf import FPDF
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

# ---------------------------
# Initialization & Config
# ---------------------------
st.set_page_config(page_title="AI-Powered Legal Document Analysis", layout="wide")
load_dotenv()  # Load environment variables

# Initialize Groq client with your API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# Initialize embedding model (you can choose another if needed)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a system prompt for legal document summarization and risk analysis
system_prompt = (
    "You are an AI assistant specialized in summarizing legal documents. Your task is to extract key points, risks, obligations, "
    "and important clauses from legal documents. Provide a concise, professional summary with minimal stopwords and no repeated words. "
    "When assessing risks, provide risk categories, severity, and recommendations."
)

# Ensure NLTK stopwords are downloaded
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words("english"))

# ---------------------------
# Session State Initialization
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "risks" not in st.session_state:
    st.session_state.risks = {}
if "ai_risk" not in st.session_state:
    st.session_state.ai_risk = ""
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "comparison" not in st.session_state:
    st.session_state.comparison = ""

# ---------------------------
# Utility Functions
# ---------------------------
def extract_text_from_pdf(file):
    """Extract text from a PDF file using PyMuPDF (fitz)."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def split_text_into_chunks(text, chunk_size=1000):
    """Split text into chunks of given size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def generate_embeddings(chunks):
    """Generate embeddings for a list of text chunks."""
    embeddings = embedding_model.encode(chunks)
    return np.array(embeddings)

def build_faiss_index(embeddings):
    """Build a FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_relevant_chunks(query, index, text_chunks, k=3):
    """Retrieve the top-k most relevant chunks using FAISS."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    return [text_chunks[i] for i in indices[0]]

def get_groq_response(input_text, temperature=0.7, top_p=0.9, max_retries=3):
    """Call the Groq API to get a response based on the input text with specified settings."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gemma2-9b-it",  # Use the Groq model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=temperature,
                max_tokens=1024,
                top_p=top_p,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(2)
                continue
            return f"Error from Groq API: {str(e)}"

def clean_summary(text):
    """Remove punctuation and stopwords, and remove duplicate words while preserving order."""
    translator = str.maketrans('', '', string.punctuation)
    words = text.translate(translator).split()
    seen = set()
    filtered = []
    for word in words:
        lower = word.lower()
        if lower in STOPWORDS:
            continue
        if lower not in seen:
            seen.add(lower)
            filtered.append(word)
    return " ".join(filtered)

def rag_summarize(text, chunk_size=1000):
    """Perform map-reduce summarization on large texts using the Groq API."""
    chunks = split_text_into_chunks(text, chunk_size)
    summaries = []
    for chunk in chunks:
        prompt = system_prompt + "\nSummarize the following segment concisely with minimal stopwords and no repeated words:\n" + chunk
        summary_chunk = get_groq_response(prompt, temperature=0.7, top_p=0.9)
        summary_chunk = clean_summary(summary_chunk)
        summaries.append(summary_chunk)
    combined = " ".join(summaries)
    final_prompt = system_prompt + "\nSummarize the following combined summary to produce a final concise summary:\n" + combined
    final_summary = get_groq_response(final_prompt, temperature=0.7, top_p=0.9)
    final_summary = clean_summary(final_summary)
    return final_summary

def detect_risks(text):
    """Basic keyword-based risk detection counting occurrences."""
    risk_categories = {
        "Compliance Risks": ["compliance", "regulation", "legal requirement"],
        "Financial Risks": ["financial loss", "penalty", "liability"],
        "Operational Risks": ["operational failure", "breach", "disruption"],
        "Termination Risks": ["termination", "cancel", "expire"]
    }
    risks = {}
    lower_text = text.lower()
    for category, keywords in risk_categories.items():
        risks[category] = sum(lower_text.count(keyword) for keyword in keywords)
    return risks

def rag_risk_assessment(text):
    """Perform an AI-driven risk assessment using retrieval augmented generation (RAG)."""
    if st.session_state.faiss_index and st.session_state.text_chunks:
        relevant = retrieve_relevant_chunks("risk compliance legal penalty liability termination", st.session_state.faiss_index, st.session_state.text_chunks, k=5)
        context = "\n".join(relevant)
        prompt = system_prompt + "\nBased on the following context, provide an AI-driven risk assessment including risk categories, severity, and recommendations:\n" + context
        risk_assessment = get_groq_response(prompt, temperature=0.7, top_p=0.9)
        return risk_assessment
    else:
        return "No AI-driven risk analysis available."

@st.cache_data(show_spinner=False)
def fetch_compliance_guidelines():
    """Fetch compliance guidelines from publicly available websites (no permission required)."""
    urls = {
        "GDPR": "https://gdpr.eu/",
        "HIPAA": "https://www.hhs.gov/hipaa/index.html"
    }
    guidelines = {}
    for key, url in urls.items():
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = soup.find_all('p')
                # Extract text from the first few paragraphs
                text_content = " ".join(p.get_text() for p in paragraphs[:5])
                guidelines[key] = text_content
            else:
                guidelines[key] = ""
        except Exception as e:
            guidelines[key] = ""
    return guidelines

def analyze_compliance(document_text):
    """Analyze document compliance by comparing it to fetched guidelines."""
    guidelines = fetch_compliance_guidelines()
    results = {}
    for key, guideline in guidelines.items():
        if guideline:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([document_text, guideline])
            sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            results[key + " Compliance Score"] = sim_score
        else:
            results[key + " Compliance Score"] = 0.0
    return results

def enhanced_compliance_analysis(document_text):
    """Provide a compliance status based on similarity scores with official guidelines."""
    scores = analyze_compliance(document_text)
    results = {}
    threshold = 0.1  # Adjust threshold as needed
    for key, score in scores.items():
        status = "Compliant" if score > threshold else "Non-Compliant"
        results[key.replace(" Compliance Score", "")] = status
        results[key + " (Score)"] = f"{score:.2f}"
    return results

def compare_documents_advanced(text1, text2):
    """Compare two documents and return a unified diff along with a cosine similarity score."""
    diff_text = "\n".join(difflib.unified_diff(text1.splitlines(), text2.splitlines(), lineterm=""))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    result = f"Document Similarity Score (Cosine Similarity): {sim_score:.2f}\n\nDifferences:\n{diff_text}"
    return result

def create_pie_chart(risk_data):
    """Generate a pie chart for risk distribution."""
    if not risk_data["Count"] or all(count == 0 for count in risk_data["Count"]):
        st.warning("No risks detected to display in the pie chart.")
        return None
    fig, ax = plt.subplots()
    ax.pie(risk_data["Count"], labels=risk_data["Risk Category"], autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    return fig

def sanitize_text(text):
    """Replace problematic Unicode characters with ASCII equivalents and remove any that cannot be encoded in latin1."""
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\uf0b7": "-"  # Replace bullet points with a hyphen.
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    return text.encode("latin1", errors="ignore").decode("latin1")

def generate_pdf_report(summary, risks, compliance, comparison):
    """Generate a PDF report combining summary, risk assessment, compliance check, and document comparison."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(40, 10, "Legal Document Analysis Report")
    pdf.ln(10)
    
    # Summary Section
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "Summary:")
    pdf.ln(8)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 10, sanitize_text(summary))
    
    # Basic Risk Assessment Section
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "Basic Risk Assessment:")
    pdf.ln(8)
    pdf.set_font("Arial", "", 10)
    for category, count in risks.items():
        pdf.cell(0, 10, f"{sanitize_text(category)}: {count}", ln=True)
    
    # AI-Driven Risk Assessment Section
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "AI-Driven Risk Assessment:")
    pdf.ln(8)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 10, sanitize_text(st.session_state.ai_risk))
    
    # Compliance Check Section (Using enhanced compliance analysis)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "Compliance Check:")
    pdf.ln(8)
    pdf.set_font("Arial", "", 10)
    for key, val in compliance.items():
        pdf.cell(0, 10, f"{sanitize_text(key)}: {sanitize_text(val)}", ln=True)
    
    # Document Comparison Section
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(40, 10, "Document Comparison:")
    pdf.ln(8)
    pdf.set_font("Arial", "", 8)
    pdf.multi_cell(0, 5, sanitize_text(comparison))
    
    return pdf

def send_email_with_report(email, pdf):
    """Send the generated PDF report via email using SendGrid."""
    try:
        sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
        pdf_output = pdf.output(dest="S").encode("latin1")
        encoded_file = base64.b64encode(pdf_output).decode()
        message = Mail(
            from_email=os.getenv("FROM_EMAIL"),
            to_emails=email,
            subject="Your Legal Document Analysis Report",
            html_content="<p>Please find attached the PDF report for your legal document analysis.</p>"
        )
        attachedFile = Attachment(
            FileContent(encoded_file),
            FileName("Legal_Report.pdf"),
            FileType("application/pdf"),
            Disposition("attachment")
        )
        message.attachment = attachedFile
        response = sg.send(message)
        return response.status_code
    except Exception as e:
        return str(e)

# ---------------------------
# Main UI with Tabs
# ---------------------------
st.title("AI-Powered Legal Document Summarization and Risk Identification")

tabs = st.tabs([
    "Upload & Summarization", 
    "Risk Assessment", 
    "Chatbot & Retrieval", 
    "Compliance & Comparison", 
    "PDF Report Generation", 
    "Email Integration"
])

# ----- Tab 1: Upload & Summarization -----
with tabs[0]:
    st.header("Upload Legal Document and Generate Summary")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                # Extract text using PyMuPDF
                text = extract_text_from_pdf(uploaded_file)
                st.session_state.full_text = text
                # Split text into chunks and build FAISS index
                chunks = split_text_into_chunks(text)
                st.session_state.text_chunks = chunks
                embeddings = generate_embeddings(chunks)
                index = build_faiss_index(embeddings)
                st.session_state.faiss_index = index
                # Generate final summary using RAG summarization
                final_summary = rag_summarize(text)
                st.session_state.summary = final_summary
                # Basic keyword risk detection
                risks_basic = detect_risks(text)
                st.session_state.risks = risks_basic
                # AI-driven risk assessment (RAG-based)
                ai_risk = rag_risk_assessment(text)
                st.session_state.ai_risk = ai_risk
                st.success("Document processed successfully!")
                st.subheader("Clean Summary")
                st.write(final_summary)
            except Exception as e:
                st.error(f"Error processing document: {e}")

# ----- Tab 2: Risk Assessment -----
with tabs[1]:
    st.header("Risk Assessment")
    if st.session_state.risks:
        st.subheader("Basic Detected Risks")
        for cat, count in st.session_state.risks.items():
            st.metric(label=cat, value=count)
        risk_data = {
            "Risk Category": list(st.session_state.risks.keys()),
            "Count": list(st.session_state.risks.values())
        }
        fig = create_pie_chart(risk_data)
        if fig:
            st.pyplot(fig)
        st.markdown("---")
        st.subheader("AI-Driven Risk Assessment")
        st.write(st.session_state.ai_risk)
    else:
        st.info("Please upload and process a document in the 'Upload & Summarization' tab first.")

# ----- Tab 3: Chatbot & Retrieval -----
with tabs[2]:
    st.header("Chatbot & Document Retrieval")
    user_query = st.text_input("Enter your legal query:")
    if st.button("Ask", key="chatbot"):
        if st.session_state.faiss_index and st.session_state.text_chunks:
            relevant = retrieve_relevant_chunks(user_query, st.session_state.faiss_index, st.session_state.text_chunks)
            context = "\n".join(relevant)
            query_prompt = system_prompt + "\nContext:\n" + context + "\nQuestion: " + user_query
            response = get_groq_response(query_prompt, temperature=0.7, top_p=0.9)
            st.write("Response:")
            st.write(response)
        else:
            st.info("Please process a document first in the 'Upload & Summarization' tab.")

# ----- Tab 4: Compliance & Comparison -----
with tabs[3]:
    st.header("Compliance Check and Document Comparison")
    if st.session_state.full_text:
        # Use the enhanced compliance analysis that fetches guidelines and analyzes the document.
        compliance_results = enhanced_compliance_analysis(st.session_state.full_text)
        st.subheader("Compliance Check Results")
        for key, val in compliance_results.items():
            st.write(f"{key}: {val}")
    else:
        st.info("Please upload and process a document in the 'Upload & Summarization' tab.")
    
    st.markdown("---")
    st.subheader("Document Comparison")
    st.write("Upload a second document to compare with the processed document.")
    compare_file = st.file_uploader("Upload second PDF document for comparison", type=["pdf"], key="compare")
    if compare_file and st.session_state.full_text:
        try:
            text2 = extract_text_from_pdf(compare_file)
            comparison_result = compare_documents_advanced(st.session_state.full_text, text2)
            st.session_state.comparison = comparison_result
            st.text_area("Comparison Result", comparison_result, height=300)
        except Exception as e:
            st.error(f"Error comparing documents: {e}")

# ----- Tab 5: PDF Report Generation -----
with tabs[4]:
    st.header("Generate PDF Report")
    if st.session_state.summary and st.session_state.risks and st.session_state.full_text:
        # Use the enhanced compliance analysis for the report
        compliance_results = enhanced_compliance_analysis(st.session_state.full_text)
        comparison_text = st.session_state.get("comparison", "No comparison document uploaded.")
        pdf = generate_pdf_report(st.session_state.summary, st.session_state.risks, compliance_results, comparison_text)
        pdf_output = pdf.output(dest="S").encode("latin1")
        st.download_button("Download PDF Report", data=pdf_output, file_name="Legal_Report.pdf", mime="application/pdf")
    else:
        st.info("Please upload and process a document to generate the report.")

# ----- Tab 6: Email Integration -----
with tabs[5]:
    st.header("Send Report via Email")
    email = st.text_input("Enter your email address:")
    if st.button("Send Email", key="send_email"):
        if email and st.session_state.summary:
            compliance_results = enhanced_compliance_analysis(st.session_state.full_text)
            comparison_text = st.session_state.get("comparison", "No comparison document uploaded.")
            pdf = generate_pdf_report(st.session_state.summary, st.session_state.risks, compliance_results, comparison_text)
            status = send_email_with_report(email, pdf)
            if str(status) == "202":
                st.success(f"Email sent successfully to {email}!")
            else:
                st.error(f"Failed to send email: {status}")
        else:
            st.error("Please provide a valid email address and ensure a document is processed.")
