import tempfile
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader
)
from typing import BinaryIO,List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_core.documents import Document
from app.utils.logger import get_logger
from app.config import Settings

logger=get_logger(__name__)

class DocumentProcessor:
   
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}

    def __init__(self,chunk_size:int|None=None,chunk_overlap:int|None=None):

        settings=Settings()
        self.chunk_size=chunk_size or settings.chunk_size
        self.chunk_overlap=chunk_overlap or settings.chunk_overlap

        self.textsplitter=RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )

        logger.info(f"Document processor initialized with chunk_size={self.chunk_size} and chunk_overlap={self.chunk_overlap}")

    
    def load_pdf(self,file_path:str|Path)->list[Document]:

        file_path=Path(file_path)
        logger.info(f"Loading PDF: {file_path.name}")
        loader=PyPDFLoader(file_path=str(file_path))
        documents=loader.load()

        logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
        return documents
    
    def load_text(self,file_path:str|Document)->list[Document]:
        file_path=Path(file_path)
        logger.info(f"Loading Text: {file_path.name}")
        loader=TextLoader(file_path=str(file_path))
        documents=loader.load()

        logger.info(f"Loaded Text file {file_path.name}")
        return documents
    
    def load_csv(self,file_path:str|Document)->list[Document]:
        file_path=Path(file_path)
        logger.info(f"Loading CSV: {file_path.name}")
        loader=CSVLoader(file_path=str(file_path))
        documents=loader.load()

        logger.info(f"Loaded {len(documents)} rows from {file_path.name}")
        return documents
    
    def load_file(self,file_path:str|Path)->list[Document]:
        file_path=Path(file_path)
        extension=Path(file_path).suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported File type .Supported filetypes are {self.SUPPORTED_EXTENSIONS}")
        loaders={
             ".pdf": self.load_pdf,
            ".txt": self.load_text,
            ".csv": self.load_csv,
        }
        loaders[extension](str(file_path))

    def load_from_upload(self,file:BinaryIO,filename:str)->list[Document]:
        extension=Path(filename).suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported File type .Supported filetypes are {self.SUPPORTED_EXTENSIONS}")
        
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:
            temp_file.write(file.read())
            temp_path=temp_file.name

        try:
            documents=self.load_file(temp_path)
            for doc in documents:
                doc.metadata["source"]=filename

            return documents
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def split_documents(self,documents:list[Document])->list[Document]:

        logger.info(f"Splitting {len(documents)} documents into chunks")
        chunks=self.textsplitter.split_documents(documents)

        return chunks
    def process_file(self,file_path:str|Path)->list[Document]:
        file_path=Path(file_path)
        documents=self.load_file(str(file_path))
        chunks=self.split_documents(documents)

    def process_upload(self,file:BinaryIO,filename:str)->list[Document]:
        documents=self.load_from_upload(file,filename)
        return self.split_documents(documents)



