from langchain.document_loaders import PyPDFLoader,TextLoader,UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from chromadb.config import Settings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
import tiktoken
import chromadb
import os
from dotenv import load_dotenv
load_dotenv()


persist_directory = 'local-chromadb'
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
encoding = tiktoken.encoding_for_model('davinci')
tokenizer = tiktoken.get_encoding(encoding.name)
client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="local-chromadb"
))


def tk_len(text):
    token = tokenizer.encode (
        text,
        disallowed_special=()
    )
    return len(token)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=tk_len,
    separators=['\n\n','\n',',','']
)

def Embeddings(chunks, collection_name):
    vectordb = Chroma.from_documents(chunks,embeddings,persist_directory=persist_directory, collection_name=collection_name)
    vectordb.persist()


def saveFiles(text, collection_name):
    chunks = text_splitter.split_text(text)
    docs = [Document(page_content=t) for t in chunks]
    print('chunky',len(docs))
    Embeddings(docs, collection_name)
    return 'success'




def Delete_files(filename):
    metadata = {}
   
    collection_name = client.list_collections()[0].name
    collection = client.get_collection(name=collection_name)
    metadata['source'] = filename
    collection.delete(
        where=metadata
    )    


def response(query, collection_name):
    print(client.list_collections())
    vectordb = Chroma(embedding_function=embeddings, persist_directory='local-chromadb', collection_name=collection_name)
    assist = RetrievalQA.from_llm(ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo"),
                                                retriever=vectordb.as_retriever(kwargs={'10'}))
    response = assist(query)
    return response['result']

