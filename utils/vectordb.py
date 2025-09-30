import os
import re
import requests
from typing import List, Tuple
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


class PrepareVectorDB:
    """
    A class for preparing and saving a VectorDB using OpenAI embeddings.
    """

    def __init__(
            self,
            url: str,
            persist_directory: str,
            embedding_model_engine: str,
            chunk_size: int,
            chunk_overlap: int
    ) -> None:
        """
        Initialize the PrepareVectorDB instance.
        """
        self.log = ''
        self.embedding_model_engine = embedding_model_engine
        self.url = url
        self.persist_directory = persist_directory

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embedding = OpenAIEmbeddings()

    def load_google_doc(self, url: str) -> Tuple[str, str]:
        """
        Load a single Google Docs document.
        """
        # Extract Google document ID from URL using regular expressions
        match_ = re.search('/document/d/([a-zA-Z0-9-_]+)', url)

        # If ID not found - generate exception
        if match_ is None:
            raise ValueError('Invalid Google Docs URL')

        # First element in search result
        doc_id = match_.group(1)

        # Download Google document by its ID in text format
        response = requests.get(
            f'https://docs.google.com/document/d/{doc_id}/export?format=txt',
            timeout=30
        )

        # For unsuccessful request statuses, an exception will be raised
        response.raise_for_status()

        # Extract data as text
        text = response.text

        return text, doc_id

    def __load_google_doc(self, url: str) -> List[Document]:
        """
        Load a Google Docs document and convert to Document format.
        """
        doc_counter = 0
        self.log += "Loading Google Docs document...\n"
        docs = []

        try:
            doc_content, doc_id = self.load_google_doc(url)
            google_doc = Document(
                page_content=doc_content,
                metadata={
                    "source": url,
                    "document_type": "google_doc",
                    "doc_id": doc_id
                }
            )
            docs.append(google_doc)
            doc_counter = 1
            self.log += f"Successfully loaded Google Docs document\n"

        except Exception as e:
            self.log += f"Error loading Google Doc: {str(e)}\n"
            return []  # Return empty list on error

        return docs

    def __chunk_documents(self, docs: List) -> List:
        """
        Chunk the loaded documents using the specified text splitter.
        """
        if not docs:
            self.log += "No documents to chunk\n"
            return []

        self.log += f"Chunking documents...\n"
        chunked_documents = self.text_splitter.split_documents(docs)
        self.log += f"Number of chunks: {len(chunked_documents)} \n\n"
        return chunked_documents

    def prepare_and_save_vectordb(self) -> Chroma:
        """
        Load, chunk, and create a VectorDB with OpenAI embeddings, and save it.
        """
        docs = self.__load_google_doc(self.url)

        if not docs:
            self.log += "No documents loaded, cannot create VectorDB\n"
            raise ValueError("No documents were successfully loaded")

        chunked_documents = self.__chunk_documents(docs)

        if not chunked_documents:
            self.log += "No chunks created, cannot create VectorDB\n"
            raise ValueError("No document chunks were created")

        self.log += f"Preparing vectordb...\n"
        vectordb = Chroma.from_documents(
            documents=chunked_documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        self.log += f"VectorDB is created and saved.\n"
        self.log += f"Number of vectors in vectordb: {vectordb._collection.count()} \n\n"
        return vectordb