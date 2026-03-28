#动态模型切换[Agents - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/agents)
from textwrap import dedent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import os
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basemodel = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3.5-plus",
    temperature=0.7,
    timeout=200,
    max_completion_tokens=1000,
)
advanced_model = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen3-max",
    temperature=0.7,
    timeout=200,
    max_completion_tokens=1000,
)


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """
    动态选择模型
    """
    message_count = len(request.state["messages"])
    if message_count > 1:
        model = advanced_model
    else:
        model = basemodel
    print(f"Selected model: {model.model_name}")
    return handler(request.override(model=model))


SYSTEM_PROMPT = dedent(
    f"""
你是一个智能助手，你的任务是回答用户关于UE(虚幻引擎)的问题。
"""
).strip()

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

agent = create_agent(
    model=basemodel,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    middleware=[dynamic_model_selection],
)

print("First invocation:")
response = agent.invoke(
    {"messages": [{"role": "user", "content": "什么是Nanite？"}]},
    config=config,
)
print(response["messages"])
#time.sleep(1)
print("Second invocation:")
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Nanite适合多少面数以上的模型？"}]},
    config=config,
)
print(response["messages"])
