from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-20b")
response = llm.invoke("Hello, ek line mein bata AI kya hai")
print(response.content)
