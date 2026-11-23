from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.agents import create_agent

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

search_tool = TavilySearch(max=2)

#  list of tools kyunki ise hum langchain ke andar pass karenge 
tools = [search_tool]

# 4. Latest LangChain v1.x Single AI Agent banana
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful AI assistant. Always use the search tool for current or real-time information.",
)

# 5. Agent ko ek Real-Time sawaal dena (Video Demo)
query = "Tell me the latest news about AI in 2026."
print(f"🧑 User Query: {query}\n")

response = agent.invoke({"messages": [{"role": "user", "content": query}]})


# 6. Andar ke steps (verbose=True jaisa) print karna
print("--- 🔍 Agent Execution Trace (Steps) ---")

for msg in response["messages"]:
    print(f"[{msg.type.upper()}]: {msg.content[:200] if msg.content else msg.tool_calls}\n")

print("--- ✅ Final Answer ---")
print(response["messages"][-1].content) # -1 isiliye kiya kyunki messages mein last message AI message hota hain , sytem message, human message, AI message 
