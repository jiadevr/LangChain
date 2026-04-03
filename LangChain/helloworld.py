#example URL：[LangChain overview - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/overview)
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
import os


@tool
def get_weather_for_location(city: str) -> str:
    """获取指定地区天气"""
    return f"{city}现在天气为 小雪!"


chatLLM = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,
)

# 小米mimo-v2-pro
# chatLLM = ChatOpenAI(
#     api_key=os.getenv("MIMO_API_KEY"),
#     base_url="https://api.xiaomimimo.com/v1",
#     model="mimo-v2-pro",
#     temperature=0.7,
# )

agent = create_agent(
    model=chatLLM,
    tools=[
        get_weather_for_location,
    ],
)
messages = [
    SystemMessage(content="你是一个智能助手，你可以使用工具回答用户的问题"),
    HumanMessage(content="今天北京天气怎么样？"),
]

agent_input: dict = {"messages": messages}
response = agent.invoke(agent_input)

print(response)
