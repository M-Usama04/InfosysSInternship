# Legal Document Summarization and Risk Assessment System

This is an AI-powered web application developed using **Streamlit**, designed to summarize legal documents, assess risks, and ensure compliance with regulations like **GDPR** and **HIPAA**. The system leverages NLP techniques for document processing and advanced summarization models for efficient legal document analysis.

## Features

### 1. **Document Summarization**:
   - Summarizes large legal documents into concise, readable summaries, helping users quickly understand the content.
   - Supports multi-document summarization and comparison.


### 2. **Risk Assessment**:
   - Identifies potential risks and compliance violations in legal documents.
   - Flagging areas of concern such as data protection, security, and more.

### 3. **Document Compliance Integration**:
   - **GDPR & HIPAA Compliance**: Analyzes documents to ensure they meet the necessary regulatory standards, such as data protection laws.
   - Provides real-time reports on whether documents comply with GDPR, HIPAA, or other relevant regulations.

### 4. **Document Comparison**:
   - Compares multiple legal documents to find differences and similarities.
   - Useful for identifying amendments, revisions, or inconsistencies.

### 5. **Email Integration**:
   - Sends summarized legal documents via email.
   - Integration with **SendGrid** for email delivery.

### 7. **User-Friendly Interface**:
   - Built using **Streamlit** for a seamless web interface.
   - Users can upload documents, receive summarized versions, check compliance, and more.

---

## UI Description

The user interface of the application is designed for simplicity and effectiveness:

1. **Home Page**:
   - Allows users to upload legal documents (PDFs, JPG, JPEG).
   - Displays a progress bar while processing the document.
   
2. **Summarization Section**:
   - Shows the generated summary in two formats: **short** and **long**.
   - Allows users to download the summary in **PDF** format.
   
3. **Compliance Check Section**:
   - Displays the compliance status for GDPR and HIPAA regulations.
   - Provides detailed risk flags and suggestions for corrections.

4. **Comparison Section**:
   - Displays the side-by-side comparison of two legal documents, highlighting the differences.

5. **Email Sending Section**:
   - Allows users to send the summarized documents via email.

