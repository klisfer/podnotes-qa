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
from flask_cors import CORS
import subprocess
from dotenv import load_dotenv
from pytube import YouTube
from moviepy.editor import AudioFileClip
import mimetypes
load_dotenv()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024

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
    if 'file' in request.files :
         file = request.files['file']
         userEmail = request.form.get("userEmail")

         if file.filename == '':
            return 'No selected file', 400
         if file:
            filename = secure_filename(file.filename)
            print(os.path.join('/workspace', filename), userEmail)
            file.save(os.path.join('/workspace', filename))
            
            file_content = parse_file(os.path.join('/workspace', filename))
            
            print(file_content)
            summary = textSummarisation.summarize_large_text_langchain(file_content, 'workspace/summary.md')
            print("uploaded summary", summary)
            save_db_results= await DBFunctions.save_summary(file_content, summary, userEmail)
            print(save_db_results)
    return summary

        

@app.route("/summarise-media", methods=['GET'])
async def summarise_media_url():

    # get url from query params and download media
    media_url = request.args.get("url")
    userEmail = request.args.get("userEmail")
    print("url is", media_url)
    if 'youtube' in media_url:
        download_video(media_url)
    else:
        audio_file = requests.get(media_url)
        print(audio_file.status_code,  userEmail)
        if audio_file.status_code == 200:
            with open('workspace/media.mp3', "wb") as file:
                file.write(audio_file.content)

    print('file downloaded')

    # delete transcripts if it exists 
    delete_if_exists('workspace/media.txt')
    delete_if_exists('workspace/media.ts.txt')
    # transcribe audio using powershell script
    try:
        result = subprocess.run(
            ["powershell.exe", "-ExecutionPolicy", "Bypass", "-File", "api/Scripts/transcribe.ps1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        print("Output:", result)
    except subprocess.CalledProcessError as error:
        print(f"Error occurred: {error}")


    #load transcript and summarise text and save to firestore
    text_summary = ''
    with open('workspace/media.txt', 'r') as file:
        content = file.read()
        text_summary = textSummarisation.summarize_large_text(content, 'workspace/summary.md')
    
    save_db_results= await DBFunctions.save_summary(content, text_summary, userEmail)

    return text_summary  
  
@app.route("/summarise-media", methods=['POST'])
async def summarise_media_upload():

    # get url from query params and download media
    userEmail = request.args.get("userEmail")

    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('workspace', filename)

        mimetype = mimetypes.guess_type(filepath)[0]

        if mimetype and mimetype.startswith('video'):
            file.save(filepath)
            audio_filename = 'media' + '.mp3'
            audio_filepath = os.path.join(app.config['workspace'], audio_filename)

            audioclip = AudioFileClip(filepath)
            audioclip.write_audiofile(audio_filepath)
            os.remove(filepath)  # delete the original video file
            print('file downloaded')

              
        elif mimetype and mimetype.startswith('audio'):
            path = os.path.join('workspace', 'media.mp3')
            file.save(path)


        # delete transcripts if it exists 
        delete_if_exists('workspace/media.txt')
        delete_if_exists('workspace/media.ts.txt')
        # transcribe audio using powershell script
        try:
            result = subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", "api/Scripts/transcribe.ps1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            print("Output:", result)
        except subprocess.CalledProcessError as error:
            print(f"Error occurred: {error}")


        #load transcript and summarise text and save to firestore
        text_summary = ''
        with open('workspace/media.txt', 'r') as file:
            content = file.read()
            text_summary = textSummarisation.summarize_large_text(content, 'workspace/summary.md')
        
        save_db_results= await DBFunctions.save_summary(content, text_summary, userEmail)
        
        return text_summary , 200

  
  

# Util functions
#  =================================================================
#  =================================================================
def delete_if_exists(file_path):
    """Delete the file at `file_path` if it exists."""
    if os.path.isfile(file_path):
        os.remove(file_path)
        


def parse_file(filepath):
    file_extension = os.path.splitext(filepath)[1]
    
    if file_extension == '.txt':
        with open(filepath, 'r') as f:
            return f.read()
    elif file_extension == '.pdf':
        with open(filepath, 'rb') as f:
            pdf = PdfReader(f)
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
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


def download_video(video_url):
    video = YouTube(video_url)
    stream = video.streams.get_lowest_resolution()
    filename = 'workspace/media'
    stream.download(filename=filename)
    print('downloaded video, converting to audio')
    # convert video to mp3 using moviepy
    video_clip = AudioFileClip(filename + '.mp4')
    video_clip.to_audiofile(filename + '.mp3')
    video_clip.close()
    print('saved audio')
 
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="0.0.0.0",port=5000)
