from flask import Flask, request
import chromaLocal
import scrapeUrl
from TextSummarisation import textSummarisation
from PyPDF2 import PdfReader
from io import BytesIO
from Utils import DBFunctions
import requests
import os
from docx import Document
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'



@app.route("/create-vector-db", methods=['GET'])
async def create_vector_collection():
    """Function to summarise transcript"""
    query_podcast = request.args.get("podcast")

    all_transcripts = DBFunctions.getAllTranscripts(query_podcast)
    text = ''
    
    for file in all_transcripts:
        contents = read_file_contents(file)
        text += "\n" + contents
        
    
    response = chromaLocal.saveFiles(text, query_podcast)
    # response = vectordb.query_response(query)
    return response


@app.route("/get-query-response", methods=['GET'])
async def get_query_response():
    """Function to summarise transcript"""        
    query = request.args.get("query")
    podcast_collection = request.args.get("podcast")

    response = chromaLocal.response(query, podcast_collection)
    return response


@app.route("/summarise-text", methods=['GET'])
async def summarise():
    """Function to summarise transcript"""
    # text = request.args.get("text")

    summary = ''
    if 'url' in request.args :
        url = request.args.get("url")
        userEmail = request.args.get("userEmail")
        content = ''
        print(url, userEmail)
        if url.lower().endswith('.pdf') is True:
            content = await load_pdf(url)
            summary = textSummarisation.summarize_large_text(content, 'workspace/summary.md')
        else:
            scrapedText = scrapeUrl.scrape_url(url)
            print('scrapedText',scrapedText)
            content = scrapedText
            summary = textSummarisation.summarize_large_text(content, 'workspace/summary.md')
        save_db_results= await DBFunctions.save_summary(content, summary, userEmail)
        print(save_db_results)
 
        
    # with open('workspace/episode.txt', 'r') as file:
    #     contents = file.read()
    #     summary = textSummarisation.summarize_large_text(contents.replace('\n',''), 'workspace/summary.txt')
    return summary

@app.route("/summarise-text", methods=['POST'])
async def summarise_upload():
    """Function to summarise transcript"""
    # text = request.args.get("text")

    summary = ''
    if 'file' in request.args :
         file = request.files['file']
         if file.filename == '':
            return 'No selected file', 400
         if file:
            filename = secure_filename(file.filename)
            print(os.path.join('/tmp', filename))
            file.save(os.path.join('/tmp', filename))
            
            file_content = parse_file(os.path.join('/tmp', filename))
            
            print(file_content)
            
            return 'File uploaded successfully', 200
 
        
    # with open('workspace/episode.txt', 'r') as file:
    #     contents = file.read()
    #     summary = textSummarisation.summarize_large_text(contents.replace('\n',''), 'workspace/summary.txt')
    return summary

def parse_file(filepath):
    file_extension = os.path.splitext(filepath)[1]
    
    if file_extension == '.txt':
        with open(filepath, 'r') as f:
            return f.read()
    elif file_extension == '.pdf':
        with open(filepath, 'rb') as f:
            pdf = PdfReader(f)
            text = ""
            for page in range(pdf.getNumPages()):
                text += pdf.getPage(page).extractText()
            return text
    elif file_extension in ['.doc', '.docx']:
        doc = Document(filepath)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    else:
        return "Unsupported file type"
    
async def load_pdf(url):
    response = requests.get(url)

    response.raise_for_status()
    pdf_file = BytesIO(response.content)
    pdf_reader = PdfReader(pdf_file)
    content = ""
    # number _of_pages = len(pdf_reader.pages)  # Use len(pdf.pages) instead of pdf.getNumPages()
    for page in pdf_reader.pages:
        content += page.extract_text()

    print('pdf_content',content)
    return content

def read_file_contents(blob):
    return blob.download_as_text()


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="0.0.0.0",port=5000)
