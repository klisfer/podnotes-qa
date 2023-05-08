from flask import Flask, request
import chromaLocal
from firebase_admin import credentials, firestore, initialize_app, storage
from dotenv import load_dotenv
load_dotenv()


cred = credentials.Certificate('./podnotes-ai-gcp-service-key.json')
app = initialize_app(cred, {'storageBucket': 'podnotes-ai.appspot.com'})
db = firestore.client()

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

    all_transcripts = getAllTranscripts(query_podcast)
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




def read_file_contents(blob):
    return blob.download_as_text()

def getAllTranscripts(search_keyword):
    bucket = storage.bucket("podnotes-transcripts")
    transcript_files = []

    # Fetch all file blobs from Firebase Storage
    all_blobs = bucket.list_blobs()

    # Check if the blob's name contains 'huberman' and add it to the list
    for blob in all_blobs:
        if search_keyword in blob.name:
            transcript_files.append(blob)
    return transcript_files


if __name__ == "__main__":
    app.run(debug=False, use_reloader=True, port=8080)
