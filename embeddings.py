from langchain_community.embeddings import HuggingFaceEmbeddings

# Load a local embedding model (works offline after download)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text1 = "The cat sat on the mat."
text2 = "A feline rested on a rug."
text3 = "The sun is shining brightly."

# Generate embeddings
embeddings = embeddings_model.embed_documents([text1, text2, text3])

embeddings1 = embeddings[0]
embeddings2 = embeddings[1]
embeddings3 = embeddings[2]

print("Embeddings for text1:", embeddings1)
print("Embeddings for text2:", embeddings2) 
print("Embeddings for text3:", embeddings3)
print(f"Length of Embeddings for text1:{len(embeddings)}")
print(f"Dimension of embedding:{len(embeddings[0])}")