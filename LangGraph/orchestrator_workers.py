# [Creating workers in LangGraph](https://docs.langchain.com/oss/python/langgraph/workflows-agents#creating-workers-in-langgraph)
import os
import operator
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage, content
from langchain_openai import ChatOpenAI
from langgraph import graph
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send


# 报告结构化输出
class Section(TypedDict):
    name: Annotated[str, ..., "章节名称"]
    description: Annotated[str, ..., "本章内容概括"]


class ReportPlan(TypedDict):
    sections: Annotated[list[Section], ..., "每一章信息组成的数组"]


# 报告状态
class ReportState(TypedDict):
    topic: str  # Report topic
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel，使用add避免静态条件，Map_Reduce中的Reduce
    final_report: str  # Final report


# Worker状态
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


chat_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,
    timeout=200,
    max_completion_tokens=1000,
)

planner = chat_llm.with_structured_output(ReportPlan)


def orchestrator(state: ReportState):
    """
    调用LLM根据用户提供的话题生成章节标题和描述
    """
    report_sections = planner.invoke(
        [
            SystemMessage(
                content="你是一个报告生成助手，请根据用户提供的话题生成报告框架，章节数量不超过5个，每章的描述不超过100字"
            ),
            HumanMessage(content=f"话题为： {state['topic']}"),
        ]
    )

    return {"sections": report_sections["sections"]}


def workers(state: WorkerState):
    """
    批量调用根据section和描述撰写具体内容
    """
    section = chat_llm.invoke(
        [
            SystemMessage(
                content="请根据报告章节和描述生成报告概括，概括内容在300字以内，使用markdown格式输出"
            ),
            HumanMessage(
                content=f"章节名称{state['section']['name']},章节描述{state['section']['description']}"
            ),
        ]
    )
    print(f"section {state['section']['name']} finished")
    print(section.content)
    return {"completed_sections": [section.content]}


# 条件边，把章节分配给多个Worker，Map_Reduce中的Map
def assign_workers(state: ReportState):
    # 注意这里是构建**动态边**，使用Send动态派发任务，会形成动态屏障，直到全部都返回才进入下一个节点;静态边中只要有一个就进入下一个节点
    return [Send("workers", {"section": section}) for section in state['sections']]


# 合成报告
def synthesizer(state: ReportState):
    completed_sections = state['completed_sections']
    completed_report_sections = "\n\n---\n\n".join(completed_sections)
    print(f"add {len(completed_sections)} section(s)")
    return {"final_report": completed_report_sections}


graph_builder = StateGraph(ReportState)
graph_builder.add_node("orchestrator", orchestrator)
graph_builder.add_node("workers", workers)
graph_builder.add_node("synthesizer", synthesizer)

graph_builder.add_edge(START, "orchestrator")
graph_builder.add_conditional_edges("orchestrator", assign_workers, ["workers"])
graph_builder.add_edge("workers", "synthesizer")
graph_builder.add_edge("synthesizer", END)

orchestrator_workers = graph_builder.compile()

state = orchestrator_workers.invoke({"topic": "帮我写一份关于 LLM Scaling Laws的报告"})
print("________________finish________________")
print(state["final_report"])
