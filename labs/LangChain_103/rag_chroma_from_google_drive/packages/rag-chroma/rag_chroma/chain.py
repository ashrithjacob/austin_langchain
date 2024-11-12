import os
import json
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# chain_community.document_loaders import GoogleDriveLoader
from langchain_google_community.drive import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
from langchain_community.document_loaders import PyPDFLoader
from datetime import date
from pydantic import RootModel
from dotenv import load_dotenv

load_dotenv()

def get_list_of_files_and_folders(folder_id):
    list_of_files_and_folders = []
    gauth = GoogleAuth()
    # NOTE: if you are getting storage quota exceeded error, create a new service account, and give that service account permission to access the folder and replace the google_credentials.
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    drive = GoogleDrive(gauth)

    # get list of files and folders in the folder
    foldered_list = drive.ListFile(
        {"q": "'" + folder_id + "' in parents and trashed=false"}
    ).GetList()
    for file in foldered_list:
        list_of_files_and_folders.append({
            "title":file['title'],
            "mimetype":file['mimeType'],
            "id":file['id']
            })

    return list_of_files_and_folders

def create_document_from_list(list_of_files_and_folders):
    documents = []
    gauth = GoogleAuth()
    # NOTE: if you are getting storage quota exceeded error, create a new service account, and give that service account permission to access the folder and replace the google_credentials.
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
        os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    drive = GoogleDrive(gauth)

    for file in list_of_files_and_folders:
        if file["mimetype"] == "application/pdf":
            print(f"Downloading {file['title']}")
            file_download = drive.CreateFile({"id": file["id"]})
            file_download.GetContentFile(file["title"])
            loader = PyPDFLoader(file["title"])
            data = loader.load()
            documents.extend(data)

    return documents

def create_chain(folder_id):
    list_of_files_and_folders = get_list_of_files_and_folders(folder_id)
    data = create_document_from_list(list_of_files_and_folders)
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=[" ", ",", "\n"]
        )

    all_splits = text_splitter.split_documents(data)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    # RAG prompt
    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    print("here")
    # LLM
    model = ChatOpenAI()

    # RAG chain
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )

    # Add typing for input
    class Question(RootModel):
        root: str

    chain = chain.with_types(input_type=Question)
    print(chain)
    return chain

"""
if __name__ == "__main__":
    folder_id_no_access = "1wTKfvr0oIeV9QfN5nHaF1JtJIr2FpkT6"
    folder_id_access = "1bT64IIlebg-KAI_ZwxagN_BzyTB2zrp7"
    #chain = create_chain(folder_id=folder_id_access)
    chain = create_chain(folder_id=folder_id_access)
"""