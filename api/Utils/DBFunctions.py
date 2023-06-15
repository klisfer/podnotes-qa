
from firebase_admin import credentials, firestore, initialize_app, storage


cred = credentials.Certificate('./podnotes-ai-gcp-service-key.json')
app = initialize_app(cred, {'storageBucket': 'podnotes-ai.appspot.com'})
db = firestore.client()


async def save_summary(content, summary, userEmail):
    user_ref = db.collection("knowledgebase").document(userEmail)
    user_doc = user_ref.get()
    summary_context = summary['intermediate_steps']
    summary = summary['output_text']
    # read summary file
    split_summary = summary.split('\n', 1)  # split the string at the first newline character
    title = split_summary[0]  # the first line is the title
    remaining_summary = split_summary[1] if len(split_summary) > 1 else ""
    firestore_doc = [{ 'title': title, 'summary': remaining_summary, 'summary_context': summary_context, 'raw_text' : content , 'folder': 'inbox'}]
    if user_doc.exists:
        user_ref.update({'docs': firestore.ArrayUnion(firestore_doc)})
        print(f"Successfully added summary")
    else:
        user_ref.set({'docs': firestore_doc})
        print(f"entry for the current user is not available")
        print("Successfully added first summary")



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

