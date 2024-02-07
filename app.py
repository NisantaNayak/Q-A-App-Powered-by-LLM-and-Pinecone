import streamlit as st
import os
import uuid
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, PodSpec
from dotenv import load_dotenv
import openai
from langchain_community.document_loaders import PyPDFDirectoryLoader
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
# Load environment variables
load_dotenv()
document_folder = "documents/"
pdf_files = glob.glob(f"{document_folder}/*.pdf")
processed_files_list = 'processed_files.txt'
llm = ChatOpenAI()
# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "myvectordb"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, metric="cosine", dimension=1536, spec=PodSpec(environment="gcp-starter", pod_type="starter"))
index = pc.Index(index_name)






# Function to update the list of processed files
def update_processed_files(file_path, processed_files_list):
    with open(processed_files_list, "a") as file:
        file.write(file_path + "\n")


# Function to get the list of processed files
def get_processed_files(processed_files_list):
    if not os.path.exists(processed_files_list):
        return set()
    with open(processed_files_list, "r") as file:
        return set(file.read().splitlines())


def process_document(file_path):

     #print(file_path)
     loader = PyPDFLoader(file_path)
     docs = loader.load()
        # Split PDF into chunks
     text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=800,
            chunk_overlap=20
        )
     split_docs = text_splitter.split_documents(docs)

        # OpenAI Embeddings
     embed = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"), model="text-embedding-ada-002")
        # Process chunks
     batch_size = 20
     for i in range(0, len(split_docs), batch_size):
            i_end = min(i + batch_size, len(split_docs))
            batch = split_docs[i:i_end]
            ids = []
            context_array = []
            meta_data = []
            for i, row in enumerate(batch):
                ids.append(str(uuid.uuid4()))
                context_array.append(row.page_content)
                meta_data.append({
                    'source': row.metadata["source"],
                    'page': row.metadata["page"] + 1,
                    'context': row.page_content
                })

            emb_vectors = embed.embed_documents(context_array)
            to_upsert = list(zip(ids, emb_vectors, meta_data))
            index.upsert(vectors=to_upsert)
            time.sleep(2)



processed_files = get_processed_files(processed_files_list)

# Process and display each file
for file_path in pdf_files:
    file_name = os.path.basename(file_path)
    if file_name not in processed_files:
        process_document(file_path)
        update_processed_files(file_name, processed_files_list)


# Display processed files in the sidebar
st.sidebar.title("Processed Files")
for processed_file in get_processed_files(processed_files_list):
    st.sidebar.text(processed_file)

# Create Prompt based on conext
def construct_with_prompt(query):
    limit =3750 
    embed= openai.embeddings.create(input=query, model="text-embedding-ada-002")   

    query_embed = embed.data[0].embedding
    res = index.query(vector=[query_embed], top_k=5, include_metadata=True)  
    contexts = [match['metadata']['context'] for match in res['matches']]

    prompt_start = ("Answer the question based on the context below.\n\n Context:\n")   
    prompt_end = (f"\n\nQuestion: {query} \n Answer:")  


    for i in range(1, len(contexts)):   
        if len("-".join(contexts[:i]))>= limit: 
            prompt = (prompt_start + "".join(contexts[:i-1])+prompt_end)   
            break  
        elif i==len(contexts)-1:    
            prompt = (prompt_start + "".join(contexts)+prompt_end) 
            return prompt 

def main():

    st.title("Q&A App powered by LLM and Pinecone")
    query = st.text_input("Enter your query")
    if query and st.button('Search'):
        prompt_with_contexts = construct_with_prompt(query)
    
        response = openai.completions.create(
         model="davinci-002", 
         prompt=prompt_with_contexts, temperature=0, max_tokens=350, top_p=1)
        st.header("Answer:")
        st.write(response.choices[0].text)

if __name__ == "__main__":
    main()
