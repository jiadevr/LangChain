#动态工具注册-预注册过滤[Agents - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/agents#runtime-tool-registration)
from dataclasses import dataclass
import os
from textwrap import dedent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from typing import Callable

@dataclass
class CustomContext:
    """Custom runtime context schema."""
    user_role: str

@dataclass
class CustomResponseFormat:
    """Agent回复格式定义"""
    tool_count: int
    tool_name: list[str]


@tool
def public_search(fool_str:str)->str:
    """
    公共搜索工具
    Parameters:
    - fool_str: 搜索字符串
    Returns:
    - str: 搜索结果
    """
    return "get public_search res"

@tool
def private_search(fool_str:str)->str:
    """
    私有搜索工具
    Parameters:
    - fool_str: 搜索字符串
    Returns:
    - str: 搜索结果
    """
    return "get private_search res"

@tool
def advanced_search(fool_str:str)->str:
    """
    高级搜索工具
    Parameters:
    - fool_str: 搜索字符串
    Returns:
    - str: 搜索结果
    """
    return "get advanced_search res"


chat_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,
    timeout=200,
    max_completion_tokens=1000,
)

@wrap_model_call
def dynamic_tool_selection(request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """
    动态选择工具
    """
    if request.runtime is None or request.runtime.context is None:
        user_role="l1"
    else:
        user_role=request.runtime.context.user_role
    if user_role=="l3":
        pass
    elif user_role=="l2":
        tools=[t for t in request.tools if t.name!="advanced_search"]
        request=request.override(tools=tools)
    else:
        tools=[t for t in request.tools if t.name.startswith("public_")]
        request=request.override(tools=tools)
    return handler(request)

SYSTEM_PROMPT = dedent(
    f"""
你是一个搜索助手，你需要返回所有搜索工具名称和数量。
"""
).strip()

agent=create_agent(
    model=chat_llm,
    tools=[public_search, private_search, advanced_search],
    system_prompt=SYSTEM_PROMPT,
    middleware=[dynamic_tool_selection,],
    context_schema=CustomContext,
    response_format=ToolStrategy(CustomResponseFormat),
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "请告诉我可以使用的全部搜索工具名称和数量"}]},
    context=CustomContext(user_role="l3"),
)
print(response["structured_response"])
# 没有添加记忆，所以两次互不影响
response = agent.invoke(
    {"messages": [{"role": "user", "content": "请告诉我可以使用的全部搜索工具名称和数量"}]},
    context=CustomContext(user_role="l1"),
)
print(response["structured_response"])