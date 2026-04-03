# [Quickstart - Docs by LangChain](https://docs.langchain.com/oss/python/langgraph/quickstart#1-define-tools-and-model)
# 创建一个类似ReAct的工具调用流程

import os
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain.tools import tool
from typing import Literal
from IPython.display import Image, display


# LLM和ToolCall两个节点循环调用
# 首先定义状态
class CustomState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    llm_calls: int


chat_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-max",
    temperature=0,
    timeout=200,
    max_completion_tokens=1000,
)

# chatLLM = ChatOpenAI(
#     api_key=os.getenv("MIMO_API_KEY"),
#     base_url="https://api.xiaomimimo.com/v1",
#     model="mimo-v2-pro",#"mimo-v2-omni"
#     temperature=0.7,
# )


@tool
def multiply(a: int, b: int) -> int:
    """计算 `a` 乘 `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """计算 `a` 加 `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """计算 `a` 除 `b`.

    Args:
        a: First int
        b: Second int
    """
    return a *1.0 / b


@tool
def MP(a: int, b: int) -> int:
    """计算 `a` MP `b`，MP计算不满足结合律需要逐层计算

    Args:
        a: First int
        b: Second int
    """
    return a * b + b


# 直接给模型绑定工具，并创建了一个Map用于LLM返回工具名称索引工具
tools = [add, multiply, divide, MP]
tools_by_name = {tool.name: tool for tool in tools}
# LangChain 的模型对象是不可变的（immutable），绑定工具之后会返回一个信息的LLM实例，
llm_with_tool=chat_llm.bind_tools(tools)


def llm_call(state: CustomState):
    # 让LLM决定是否继续调用工具
    # 注意这个直接在节点中调用，没有加入系统提示词，state中第一条就是用户问题，所以要拼在一起
    messages = [
        SystemMessage(
            content="你是一个代数计算助手,我们定义了一种新的计算方式MP，请根据用户输入的指令调用给出的工具进行计算."
        )
    ] + state["messages"]
    # 直接更新CustomState
    return {
        "messages": [llm_with_tool.invoke(messages)],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Call Tools
def tool_node(state: CustomState):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        print(tool_call)
        tool = tools_by_name[tool_call["name"]]
        step_res = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=step_res, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: CustomState) -> Literal["tool_node", END]:
    last_messages = state["messages"][-1]
    if last_messages.tool_calls:
        return "tool_node"
    return END


agent_builder = StateGraph(CustomState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")
agent = agent_builder.compile()

messages = [HumanMessage(content="请帮我计算(4 MP 9) MP 3的结果")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()

# qwen-plus/qwen-max比较
# 在明确说明MP运算不满足结合律的前提下Qwen-plus会强行计算前者进行并行工具调用，导致结果错误
# - qwen-plus
# {'name': 'MP', 'args': {'a': 4, 'b': 9}, 'id': 'call_8d83b52d5a524262a0adef', 'type': 'tool_call'}
# {'name': 'MP', 'args': {'a': 37, 'b': 3}, 'id': 'call_e5f85ab88bd9493591e2e5', 'type': 'tool_call'}
# ================================ Human Message =================================

# 请帮我计算(4 MP 9) MP 3的结果
# ================================== Ai Message ==================================
# Tool Calls:
#   MP (call_8d83b52d5a524262a0adef)
#  Call ID: call_8d83b52d5a524262a0adef
#   Args:
#     a: 4
#     b: 9
#   MP (call_e5f85ab88bd9493591e2e5)
#  Call ID: call_e5f85ab88bd9493591e2e5
#   Args:
#     a: 37 #<=====================编了一个值想在一次调用中解决
#     b: 3
# ================================= Tool Message =================================

# 45
# ================================= Tool Message =================================

# 114
# ================================== Ai Message ==================================
#
#
# - qwen-max
# {'name': 'MP', 'args': {'a': 4, 'b': 9}, 'id': 'call_0ff521bf3e8948958a8cd4c4', 'type': 'tool_call'}
# {'name': 'MP', 'args': {'a': 45, 'b': 3}, 'id': 'call_f84a78e96c784c2fa5d8fa15', 'type': 'tool_call'}
# ================================ Human Message =================================

# 请帮我计算(4 MP 9) MP 3的结果
# ================================== Ai Message ==================================

# 我需要分步计算这个表达式，因为MP运算不满足结合律。

# 首先计算括号内的 4 MP 9：
# Tool Calls:
#   MP (call_0ff521bf3e8948958a8cd4c4)
#  Call ID: call_0ff521bf3e8948958a8cd4c4
#   Args:
#     a: 4
#     b: 9
# ================================= Tool Message =================================

# 45
# ================================== Ai Message ==================================

# 现在我有了括号内的结果是45，接下来计算 45 MP 3：
# Tool Calls:
#   MP (call_f84a78e96c784c2fa5d8fa15)
#  Call ID: call_f84a78e96c784c2fa5d8fa15
#   Args:
#     a: 45
#     b: 3
# ================================= Tool Message =================================

# 138
# ================================== Ai Message ==================================

# 小米MinoV2-omni和pro都能正确回答