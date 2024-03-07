# !pip install langchain sentence-transformers
# !pip install unstructured

from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
import os
import shutil
import json

import os
from dotenv import load_dotenv

# Check if runpod_v1.env file exists
if os.path.exists('runpod.env'):
    # Load environment variables from runpod_v1.env file
    load_dotenv('runpod.env')
else:
    # Load environment variables from default .env file
    load_dotenv()




class EmbeddingsDB:
    def __init__(self, data_path, chroma_path, processed_files_path):
        self.data_path = data_path
        self.chroma_path = chroma_path
        self.processed_files_path = processed_files_path
        self.processed_files = self._load_processed_files()

    def _load_processed_files(self):
        if os.path.exists(self.processed_files_path):
            with open(self.processed_files_path, 'r') as file:
                return set(json.load(file))
        return set()

    def generate_data_store(self):
        documents = self.load_documents()
        chunks = self.split_text(documents)
        if chunks:  # Check if chunks list is not empty
            self.save_to_chroma(chunks)
        else:
            print("No chunks to save to Chroma.")

    def load_documents(self):
        # Collect all subdirectories in the data path recursively
        folders = self.collect_folders_recursively()
        documents = []

        # Iterate through each folder
        for folder in folders:
            try:
                # Load documents from the current folder using DirectoryLoader
                loader = DirectoryLoader(folder, glob="*.[m|t][d|x][t|]", loader_cls=TextLoader, silent_errors=True)
                # Document Structure: {'page_content': 'URL: https://blog.langchain.dev', 'metadata': {'source': 'scraped_content_v1/index.txt'}, 'type': 'Document'}
                for document in loader.load():
                    source_path = document.metadata.get('source', '')
                    with open(source_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    if source_path not in self.processed_files:
                        documents.append(Document(text, metadata={'source': source_path}))
                        self.processed_files.add(source_path)
                        print(".", end="", flush=True)  # Print a dot for each successfully loaded file
            except Exception as e:
                print(f"\nError loading documents from folder {folder}: {e}")

        print()  # Print a newline after all files are processed
        self._save_processed_files()
        return documents



    def collect_folders_recursively(self):
        folders = []
        for root, dirs, _ in os.walk(self.data_path):
            for dir in dirs:
                folders.append(os.path.join(root, dir))
        return folders

    def split_text(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def save_to_chroma(self, chunks):
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
        db = Chroma.from_documents(
            chunks, OpenAIEmbeddings(), persist_directory=self.chroma_path
        )
        db.persist()

    def _save_processed_files(self):
        with open(self.processed_files_path, 'w') as file:
            json.dump(list(self.processed_files), file)
