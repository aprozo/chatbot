import os
import arxiv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_community.document_loaders import UnstructuredMarkdownLoader

import chromadb
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore

# =============================================================================
def getArxivPaper(name):
     name = name.replace("_", "/")
     print (f"Searching for the paper with ID: {name}")
     try:
          search = arxiv.Search( id_list=[name] )
          paper = next(arxiv.Client().results(search))
          return paper
     except:
          print(f"No results found for the ID: {name}")
     return None


def getDocument(filename):
    base_filename = filename.split("/")[-1].split(".md")[0]
    paper = getArxivPaper(base_filename)
    text_metadata = {
        "title": paper.title,
        "published": paper.published.strftime('%Y-%m-%d'),
    }
    text_metadata["arxiv_id"] = paper.get_short_id()
    loader = UnstructuredMarkdownLoader(filename)
    document = loader.load()
    document[0].metadata = text_metadata
    
    print (f"Document {base_filename} has been loaded")

    return document

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20, add_start_index=True)

# choose database style
database_style = "pinecone"
# =============================================================================
# ===========================Store Document====================================

if database_style == "chroma":
    from pinecone.grpc import PineconeGRPC as Pinecone
    from pinecone import ServerlessSpec
    import os
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    import time
    index_name = "arxiv-papers-md"  # change if desired
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    from langchain_pinecone import PineconeVectorStore
    index = pc.Index(index_name)
    from uuid import uuid4
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embedding_function, )

elif database_style == "chroma":
    persistent_client = chromadb.PersistentClient(path="database")
    collection = persistent_client.get_or_create_collection("arxiv_papers")
    child_vectorstore = Chroma(
        client=persistent_client,
        embedding_function=embedding_function
    )
    # # The storage layer for the parent documents
    fs = LocalFileStore("./database/full_docs")
    parent_docstore = create_kv_docstore(fs)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    retriever = ParentDocumentRetriever(
        vectorstore=child_vectorstore, 
        docstore=parent_docstore,
        child_splitter=text_splitter,
        parent_splitter=parent_splitter)


# =============================================================================
folder_path="md"
data=[]

for filename in tqdm(os.listdir(folder_path)):
    if not filename.endswith(".md"):
         continue
    filename = os.path.join(folder_path, filename)
    print (f"Loading {filename}")

    document = getDocument(filename)
    if document is None:
        continue
    splits = text_splitter.split_documents([document[0]])
    print (f"Adding {len(splits)} chunks to the vectorstore")
    vectorstore.add_documents(documents=splits)