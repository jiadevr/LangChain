from typing import List
from langchain_core.messages import AIMessage
from langchain_core.messages import AnyMessage
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from textwrap import dedent
import os


supervisor_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,
    timeout=200,
    max_completion_tokens=1000,
)

weather_llm = supervisor_llm.model_copy(update={"model": "qwen-max"})


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city.
    Parameters:
    - city: 城市名称
    Returns:
    - str: 天气结果
    """

    return f"It's always sunny in {city}!"


weather_agent = create_agent(
    model=weather_llm,
    tools=[get_weather],
    name="weather_agent",
)


@tool
def call_weather_agent(query: str) -> str:
    """Call weather agent to get weather.
    Parameters:
    - query: 查询语句
    Returns:
    - str: 天气结果
    """
    result = weather_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].text


supervisor_agent = create_agent(
    model=supervisor_llm,
    tools=[call_weather_agent],
    name="supervisor_agent",
)

def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")


# 统一stream处理器 chunk结构为((),message,(chunk, metadata))
def handle_stream_chunk(chunk: tuple) -> None:
    if not isinstance(chunk, tuple) or len(chunk) < 2:
        return
    #print(chunk)
    _,channel, data = chunk
    if channel == "messages":
        token, metadata = data
        agent_name=metadata.get("lc_agent_name")
        print(f"{agent_name}: ")
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
    elif channel == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])

input_message = {"role": "user", "content": "北京天气如何?"}
#current_agent = None
for chunk in supervisor_agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
    subgraphs=True,
    version="v2",
):
    handle_stream_chunk(chunk)