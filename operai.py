from langchain_openai import ChatOpenAI

#Load Model
llm = ChatOpenAI(
    model="",
    temperature=0,
    api_key="",
)

#Message
response = llm.invoke("Can you explain what RAG is")
print(response.content)