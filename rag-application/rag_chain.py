from langchain.chains import RetrievalQA

# Create retriever
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# RAG pipeline: retrieval + generation
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"  # simplest: stuff retrieved docs into the prompt
)

# Ask a question
query = "What is the price of Margherita Pizza?"
answer = qa_chain.run(query)

print("Q:", query)
print("A:", answer)
