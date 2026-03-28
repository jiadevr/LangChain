# example URL：[Quickstart - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/quickstart)
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from textwrap import dedent
import os

chat_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,
    timeout=10,
    max_completion_tokens=1000,
)


@tool
def get_weather_for_location(city: str) -> str:
    """
    获取指定城市的天气情况
    Parameters:
    - city: 城市名称
    Returns:
    - str: 城市的天气情况描述
    """
    return f"{city}的天气是晴天，温度20度，湿度30%"


@dataclass
class CustomContext:
    """Custom runtime context schema."""

    user_id: str


@tool
def get_user_location(runtime: ToolRuntime[CustomContext]) -> str:
    """
    获取用户位置
    Parameters:
    - runtime: 工具运行时上下文
    Returns:
    - str: 用户所在的城市名称
    """
    user_id = runtime.context.user_id
    return "北京" if user_id == "010" else "上海"


@dataclass
class CustomResponseFormat:
    """Agent回复格式定义"""

    # 双关语回复 (总是提供)
    punny_response: str
    # 调用模型信息（总是提供）
    model_name: str
    # 其他天气相关信息 (可选)
    weather_conditions: str | None = None


SYSTEM_PROMPT = dedent(
    f"""
你是一个爱说双关语的天气预报专家，
如果用户询问你天气，你需要调用合适的工具获取用户位置，然后使用位置再次调用合适的工具获取这个位置的天气情况。
"""
).strip()
checkpointer = InMemorySaver()
llm_agent = create_agent(
    model=chat_llm,
    tools=[
        get_user_location,
        get_weather_for_location,
    ],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    context_schema=CustomContext,
    response_format=ToolStrategy(CustomResponseFormat),
    #extra_body={"enable_thinking":True}
)
config = {"configurable": {"thread_id": "1"}}
response = llm_agent.invoke(
    {"messages": [{"role": "user", "content": "现在外边天气怎么样？"}]},
    config=config,
    context=CustomContext(user_id="010"),
)
print(response["structured_response"])

response = llm_agent.invoke(
    {"messages": [{"role": "user", "content": "谢谢"}]},
    config=config,
    context=CustomContext(user_id="010"),
)
print(response["structured_response"])
