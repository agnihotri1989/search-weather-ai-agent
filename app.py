import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

# 1. Environment variables (.env) load karna
load_dotenv()

# 2. Streamlit Page Configuration (Video [01:15:25])
st.set_page_config(
    page_title="Agentic Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agentic Assistant")
st.markdown("Search and Weather Single AI Agent using **LangChain + Groq**")


# 3. Custom Weather Tool
@tool
def get_weather_data(city: str) -> str:
    """Fetches the current weather information (temperature, condition, humidity) for a given city."""
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

# 4. Agent ko cache karna taaki har button click par dobara rebuild na ho
@st.cache_resource
def build_agent():
    llm = ChatGroq(
       model="openai/gpt-oss-120b",
       temperature=0,
      )
    search_tool = TavilySearch(max_results=2)
    return create_agent(
        model=llm,
        tools=[search_tool, get_weather_data],
        system_prompt="You are a helpful AI assistant. Use search for general/current facts and get_weather_data for live weather.",
    )


agent = build_agent()

# 5. User Query Input Box & Run Button (Video [01:15:55])
user_query = st.text_input(
    "Ask me anything (e.g., Find the capital of France and its current weather):",
    placeholder="Type your question here...",
)

if st.button("Run Agent", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a query first!")
    else:
        with st.spinner("Agent is thinking..."):
            response = agent.invoke(
                {"messages": [{"role": "user", "content": user_query}]}
            )
            final_answer = response["messages"][-1].content

        st.success("Response Generated!")
        st.write(final_answer)