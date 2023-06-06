
from firebase_admin import credentials, firestore, initialize_app, storage


cred = credentials.Certificate('./podnotes-ai-gcp-service-key.json')
app = initialize_app(cred, {'storageBucket': 'podnotes-ai.appspot.com'})
db = firestore.client()


async def save_summary(summary, userEmail):
    user_ref = db.collection("knowledgebase").document(userEmail)
    user_doc = user_ref.get()
    # read summary file
  
    if user_doc.exists:
        user_ref.update({'docs': firestore.ArrayUnion([{ 'summary': summary, 'folder': 'inbox'}])})
        print(f"Successfully added summary")
    else:
        print("No such document!")
        print(f"entry for the current user is not available")



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

