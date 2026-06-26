import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz
from google import genai as modern_genai
import io
from PIL import Image
import time

load_dotenv()

# Gracefully check for API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not set. Please add GOOGLE_API_KEY to your .env file in the project folder.")
    st.stop()






def get_pdf_text(pdf_docs, status_callback=None):
    text=""
    client = modern_genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    for pdf in pdf_docs:
        pdf_bytes = pdf.getvalue()
        pdf_reader = PdfReader(pdf)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            fitz_page = doc[i]
            has_images = len(fitz_page.get_images()) > 0
            
            # Use OCR if the page contains minimal digital text or contains images/diagrams
            if not page_text or len(page_text.strip()) < 50 or has_images:
                if status_callback:
                    status_callback(f"Transcribing page {i+1} of {len(pdf_reader.pages)} using Gemini OCR...")
                
                try:
                    pix = fitz_page.get_pixmap(dpi=150)
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))
                    
                    ocr_text = ""
                    for attempt in range(3):
                        try:
                            response = client.models.generate_content(
                                model="gemini-3.1-flash-lite",
                                contents=[
                                    "Transcribe all readable text from this image page. Maintain the layout and structure as much as possible. Do not add any greetings, explanations, warnings, or formatting markers. If there is no text, reply with nothing.",
                                    image
                                ]
                            )
                            ocr_text = response.text
                            break
                        except Exception as e:
                            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                                if attempt < 2:
                                    if status_callback:
                                        status_callback(f"Rate limit hit. Waiting 10s before retry (Attempt {attempt+1}/3)...")
                                    time.sleep(10)
                                    continue
                            raise e
                    
                    if ocr_text:
                        text += ocr_text + "\n"
                except Exception as e:
                    if page_text:
                        text += page_text + "\n"
                    print(f"OCR failed for page {i+1}: {e}")
            else:
                text += page_text + "\n"
                
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.warning("No PDF index found. Please upload and process PDF files first using the sidebar menu.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001")
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error loading the index: {e}")
        return

    chain = get_conversational_chain()

    try:
        response = chain.invoke(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        print(response)
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error generating answer: {e}")




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file first.")
            else:
                status_placeholder = st.sidebar.empty()
                with st.spinner("Processing..."):
                    def update_status(msg):
                        status_placeholder.info(msg)
                    raw_text = get_pdf_text(pdf_docs, status_callback=update_status)
                    status_placeholder.empty()
                    
                    if not raw_text.strip():
                        st.error("Could not extract any text from the uploaded PDF files. Please verify the files are not scanned images or empty.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Done")



if __name__ == "__main__":
    main()