from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Load documents
loader = TextLoader(file_path="./knowledge_base/notes.txt")
documents = loader.load()

# Split into smaller chunks (e.g., 300 chars each)
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert to vectors
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in FAISS
vector_db = FAISS.from_documents(docs, embedder)

# Query
query = "What is the price of Margherita Pizza?"
results = vector_db.similarity_search(query, k=2)

for r in results:
    print(r.page_content)
