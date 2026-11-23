import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

print("--- 1. Testing Bare LLM (No Tools) ---")
llm_response = llm.invoke("Tell me a short 1-line joke about AI.")
print("LLM Answer:", llm_response.content)

# 3. Tavily Search Tool ko initialize & test karna
print("\n--- 2. Testing Tavily Search Tool (max_results=2) ---")
search_tool = TavilySearch(max_results=2)
search_result = search_tool.invoke("Give me the latest news on AI")
print("Tavily Raw Result:\n", search_result)