#动态工具注册-运行时添加[Agents - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/agents#runtime-tool-registration)
import os
from textwrap import dedent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, wrap_tool_call
# 注意这个ToolCallRequest位置不在langchain.agents.middleware
from langgraph.prebuilt.tool_node import ToolCallRequest
from langchain.messages import ToolMessage


@tool
def calculate_tip(bill_amount: float, tip_percentage: float = 20.0) -> str:
    """
    计算小费金额
    Parameters:
    - bill_amount: 订单金额，必须大于0
    - tip_percentage: 小费百分比，必须大于0（默认20%，输入20）
    Returns:
    - str: 小费金额和总金额的字符串表示
    """
    tip = bill_amount * (tip_percentage / 100)
    return f"Tip: ${tip:.2f}, Total: ${bill_amount + tip:.2f}"


@tool
def public_search(fool_str: str) -> str:
    """
    公共搜索工具
    Parameters:
    - fool_str: 搜索字符串
    Returns:
    - str: 搜索结果
    """
    return "get public_search res"

## 返回会在ToolMessage中
@wrap_tool_call
def handle_tool_errors(request, handler):
    """当工具当前不可用时调用，返回自定义错误消息"""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"],
        )


class DynamicToolMiddleware(AgentMiddleware):
    """Middleware that registers and handles dynamic tools."""

    def wrap_model_call(self, request: ModelRequest, handler):
        # Add dynamic tool to the request
        request = request.override(tools=[*request.tools, calculate_tip])
        return handler(request)

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        # Handle execution of the dynamic tool
        # 返回来的是一个字典
        if "calculate_tip" in request.tool_call.keys():
            return handler(request.override(tool=calculate_tip))
        return handler(request)


chat_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,
    timeout=200,
    max_completion_tokens=1000,
)

SYSTEM_PROMPT = dedent(
    f"""
你是一个订单助手，你需要根据用户订单金额和小费百分比计算小费金额。
必须使用给出的工具，如果工具不可用时抛出错误。
"""
).strip()

agent = create_agent(
    model=chat_llm,
    tools=[
        public_search,
    ],
    system_prompt=SYSTEM_PROMPT,
    # 注意需要穿个实例进去
    middleware=[
        DynamicToolMiddleware(),handle_tool_errors
    ],
)

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "请告诉我20%费率下订单金额85元需要付多少小费"}
        ]
    },
)
print(response)

response = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "请告诉我20%费率下订单金额-200元需要付多少小费"}
        ]
    },
)
print(response)
