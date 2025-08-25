from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import ollama
from langchain_community.vectorstores import FAISS
from langchain.chains import retrieval_qa
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader

def PyPDFLoader(filepath):
    raise NotImplementedError


#Load Documents 
#Load a pdf file
pdf_loader = PyPDFLoader("sample.pdf")
pdf_docs = pdf_loader.load()

#Load a txt file
txt_loader = TextLoader("/knowledge_base/notes.txt")
txt_docs = txt_loader.load()

#Combine pdf txt
docs = pdf_loader + txt_loader + [
    Document(page_content="LangChain is a framework for building LLM-powered applications."),
    Document(page_content="RAG stands for Retrieval-Augmented Generation."),
    Document(page_content="FAISS is a vector database for efficient similarity search."),
]

#Connection with local LLM(Mistral via Ollama)
#llm = ollama(model="mistral")
#Local embeddings
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key="")
#Store in FAISS
db = FAISS.from_documents(docs, embeddings)


#Create Retriever
retriever = db.as_retriever()

#RAG Chain (Retriever + LLM)
rag_chain = retrieval_qa.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

#Ask a Question
query = "What is RAG"
result = rag_chain.invoke(query)


print("Answer:", result["result"])
print("\nSources:")
for doc in result[""]:
    print("_", doc.page_content)