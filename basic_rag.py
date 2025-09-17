#Basic RAG implementation
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load documents
loader = TextLoader(file_path="/knowledge_base/notes.txt", )
documents = loader.load()

# Convert to vectors using HuggingFace embeddings
embedder = HuggingFaceEmbeddings()
embeddings = embedder.embed_documents([doc.page_content for doc in documents])


#Store in vector database (FAISS)
vector_db = FAISS.from_documents(documents, embedder)

# Example query
query = "What is the capital of France?"

#Result
results = vector_db.similarity_search(query)
print(results)