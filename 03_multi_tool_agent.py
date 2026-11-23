import os
import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def get_weather_data(city: str) -> str: 
    """Adds two numbers"""
    api_key = os.getenv("WEATHERSTACK_API_KEY")
    url = f"http://api.weatherstack.com/current?access_key={api_key}&query={city}"

    response = requests.get(url, timeout=10)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}."

    current = data["current"]
    temp = current.get("temperature")
    desc = ", ".join(current.get("weather_descriptions", []))
    humidity = current.get("humidity")

    return f"The current weather in {city} is {temp}°C with {desc} and {humidity}% humidity."


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)
search_tool = TavilySearch(max_results=2)

# Ab hamare Single Agent ke paas 2 tools hain!
tools = [search_tool, get_weather_data]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful AI assistant. Use search for general/current facts and get_weather_data for live weather.",
)

# 4. Multi-step query run karna (Video Demo)
query = "Find the capital of India and then find its current weather."
print(f"🧑 User Query: {query}\n")

response = agent.invoke({"messages": [{"role": "user", "content": query}]})

print("--- 🔍 Agent Execution Trace ---")
for msg in response["messages"]:
    print(f"[{msg.type.upper()}]: {msg.content[:200] if msg.content else msg.tool_calls}\n")

print("--- ✅ Final Answer ---")
print(response["messages"][-1].content)

