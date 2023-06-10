import openai
import os
import tiktoken
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from bardapi import Bard
import concurrent.futures
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

tokenizer = tiktoken.get_encoding('cl100k_base')

def count_tokens(text):
    token_count = len(tokenizer.encode(text))
    return token_count


def chunk_text(text, max_token_size):
    tokens = text.split(" ")
    token_count = 0
    chunks = []
    current_chunk = ""

    for token in tokens:
        token_count += count_tokens(token)

        if token_count <= max_token_size:
            current_chunk += token + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = token + " "
            token_count = count_tokens(token)

    if current_chunk:
        chunks.append(current_chunk.strip())
    print("chunks", len(chunks))
    return chunks

def tk_len(text):
    token = tokenizer.encode (
        text,
        disallowed_special=()
    )
    return len(token)

def split_chunks(text):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=12)
    for chunk in splitter.split_documents(text):
        chunks.append(chunk)
    return chunks

def palm_api_summary(url):
    bard = Bard()
    query = "can you summarise this link for me in 500 words, create sections subheading to explain clearly:" + url

    response = bard.get_answer(query)
    print("response", response)
    # refined_response = format_fixer(response["content"])
    return response["content"]


def format_fixer(text):
    messages = [
        {"role": "system", "content": "You are a text formatting assistant."},
        {"role": "user", "content": f"Format the following text in a markdown file format. Maintain the rest of the details of the text as it is. Add line breaks after header and bullet points end: {text}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=700,  # Adjust based on your desired summary length
        n=1,
        stop=None,
        temperature=0.1,
    )

    formatted_text = response.choices[0].message['content'].strip()
    print('summary-chunk', formatted_text)
    return formatted_text

def generate_summary(text):
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts key information from text."},
        {"role": "user", "content": f"Take notes from the text in form of bullet points (maintain the context) output atleast 200-400 words: \n\n\n{text}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,  # Adjust based on your desired summary length
        n=4,
        stop=None,
        temperature=0.1,
    )

    summary = response.choices[0].message['content'].strip()
    print('summary-chunk', summary)

    return summary


def refineSummary(text):
    token_limit = 3500
    token_length = tk_len(text)
    if token_length > token_limit:
       print('token-limit', token_limit, token_length)
       text = text[:token_limit]
       print('stripped', tk_len(text))
   


    prompt = f"this is the raw text that needs to be used to create a blog article in about 500 words. Add subheaders bullet points to make the article easily digestable. Maintain the context. Give the output in md format. Add line breaks after headers \n \n " + text
    print('refining summary', prompt)
    print('refining summary token count', tk_len(prompt))
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates blog articles based on text provided"},
            {"role": "user", "content": prompt},         
        ],  
        n=1,
        stop=None,
        temperature=0.1,
    )
    refined_summary = completion.choices[0].message.content
    print('refined summary', refined_summary)
    return refined_summary


def summarize_large_text(input_text, output_file):
    # Chunk the text into smaller parts
    input_text = input_text.replace('\n', '')
    max_token_size = 3200 
    print("max token size", max_token_size)
    text_chunks = chunk_text(input_text, max_token_size)
    # split_index = len(text_chunks) // 2
    texts = [text_chunks[i:i+4] for i in range(0, len(text_chunks), 4)]
   
    print(len(texts))
    summaries_array = []
    # # Generate summaries for each chunk concurrently
    for text_array in texts:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            summaries = list(executor.map(generate_summary, text_array))
            summaries_array.append(summaries)
            print('summaries',len(summaries))
   
    
    # # Generate summaries for each chunk
    # summaries = [generate_summary(chunk) for chunk in text_chunks]
    print('summaries',len(summaries_array))
    summaries_array = [item for sublist in summaries_array for item in sublist]

    # Combine the summaries into a single article
    article = "## Summary\n\n"
    for idx, summary in enumerate(summaries_array, 1):
        article +=  f" idx: {summary}  \n\n"

    
    refinedSummary = refineSummary(article)
    print('refined summary', refinedSummary)
    # Save the article to a Markdown file
    # with open(output_file, "w", encoding='utf-8') as f:
    #     f.write(refinedSummary)
    return refinedSummary


def summarize_large_text_langchain(input_text, output_file, max_token_size=3200):
    text_chunks = chunk_text(input_text, max_token_size)
    print('chunks', len(text_chunks))
    docs = [Document(page_content=t) for t in text_chunks]
    print('chunks', len(docs))
    prompt_template = """Summarize the following text like a section of blog article while maintaining the context of the conversation: 


    {text}


    CONCISE SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_template = (
        "This is the raw text that needs to be summarised in the form of blog article while maintaining the context. Use headers and sections:\n"
        " {existing_answer}\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    llm = ChatOpenAI(temperature=0.1, max_tokens=700, model_name="gpt-3.5-turbo", openai_api_key="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")

    chain = load_summarize_chain(llm,
        chain_type="refine", return_intermediate_steps=True, question_prompt=PROMPT, refine_prompt=refine_prompt)
    results = chain({"input_documents": docs}, return_only_outputs=True)
    print(results)

    with open(output_file, "w") as f:
        f.write(results['output_text'])


def bart_summariser(transcript):
    
    """
    It takes a transcript, tokenizes it, and then generates a summary using the BART model
    Args:
      transcript: The text you want to summarize.
    Returns:
      A summary of the text.
    """
    try:
        print("initiating summarizer...")
       
       
        tokenizer = AutoTokenizer.from_pretrained(
                "philschmid/bart-large-cnn-samsum")
        model = AutoModelForSeq2SeqLM.from_pretrained(
                "philschmid/bart-large-cnn-samsum")
        print("tokenizer and model were downloaded from huggingface")
        inputs = tokenizer(transcript,
                           max_length=1024,
                           truncation=True,
                           return_tensors="pt")
        summary_ids = model.generate(
            inputs["input_ids"], num_beams=8, min_length=200, max_length=5000)
        summary = tokenizer.batch_decode(
            summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        with open("workspace/chunks.txt", "a") as f:
            f.write(summary + "\n\n")
        print("summary generated", count_tokens(summary), summary)
        return summary
    except Exception as e:
        print("following error occured with bart summarised", e)        

