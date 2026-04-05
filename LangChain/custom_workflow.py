from typing import TypedDict, Any, cast
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
import os


class CustomState(TypedDict):
    question: str
    rewritten_question: str
    documents: list[str]
    answer: str


# 向量化
embedding_model = OpenAIEmbeddings(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="text-embedding-v3",
    # 如果API要求Str明文必须关闭这项，否则报格式错
    check_embedding_ctx_length=False,
    # Qwen的Embedding批次大小为10，单批次最大Token数量为8192
    chunk_size=10,
)

# Ollama本地Qwen3-embedding:8b
# embedding_model = OpenAIEmbeddings(
#     api_key="",
#     base_url="http://localhost:11434/v1",
#     model="qwen3-embedding:8b",
#     dimensions=1024,
#     # 如果API要求Str明文必须关闭这项，否则报格式错
#     check_embedding_ctx_length=False,
#     # Qwen的Embedding批次大小为10，单批次最大Token数量为8192
#     chunk_size=10
# )


vector_store = InMemoryVectorStore(embedding=embedding_model)
documents = [
    "New York Liberty 2024 roster: Breanna Stewart, Sabrina Ionescu, Jonquel Jones, Courtney Vandersloot.",
    "Las Vegas Aces 2024 roster: A'ja Wilson, Kelsey Plum, Jackie Young, Chelsea Gray.",
    "Indiana Fever 2024 roster: Caitlin Clark, Aliyah Boston, Kelsey Mitchell, NaLyssa Smith.",
    "2024 WNBA Finals: New York Liberty defeated Minnesota Lynx 3-2 to win the championship.",
    "June 15, 2024: Indiana Fever 85, Chicago Sky 79. Caitlin Clark had 23 points and 8 assists.",
    "August 20, 2024: Las Vegas Aces 92, Phoenix Mercury 84. A'ja Wilson scored 35 points.",
    "A'ja Wilson 2024 season stats: 26.9 PPG, 11.9 RPG, 2.6 BPG. Won MVP award.",
    "Caitlin Clark 2024 rookie stats: 19.2 PPG, 8.4 APG, 5.7 RPG. Won Rookie of the Year.",
    "Breanna Stewart 2024 stats: 20.4 PPG, 8.5 RPG, 3.5 APG.",
]
vector_store.add_texts(documents)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


@tool
def get_last_news(query: str) -> str:
    """Get the latest news about the WNBA.
    Parameters:
    - query: The query to search for.
    Returns:
    - str: The latest news about the WNBA.
    """
    return "Latest: The WNBA announced expanded playoff format for 2025..."


chat_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.7,
    timeout=200,
    max_completion_tokens=1000,
)


# #Ollama本地Gemma
# chat_llm = ChatOpenAI(
#     name="ollama-ai",
#     model="gemma4",
#     base_url="http://localhost:11434/v1",
#     api_key="",
#     temperature=0.7,
# )

agent = create_agent(
    model=chat_llm,
    tools=[get_last_news],
)


class RewrittenQuery(BaseModel):
    query: str


def rewrite_query(state: CustomState) -> dict:
    system_prompt = """Rewrite this query to retrieve relevant WNBA information.
The knowledge base contains: team rosters, game results with scores, and player statistics (PPG, RPG, APG).
Focus on specific player names, team names, or stat categories mentioned."""
    response = chat_llm.with_structured_output(RewrittenQuery).invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": state["question"]},
        ]
    )
    # 返回值： game results with scores WHERE year = 2024 AND championship = true
    print("_______________rewrite_query____________")
    print(response.query)
    return {"rewritten_question": response.query}


def retrieve_documents(state: CustomState) -> dict:
    docs = retriever.invoke(state["rewritten_question"])
    # 召回5条文档
    # [
    #     Document(
    #         id="6a432d4d-d6ea-487b-b2f0-ceead8fcdeba",
    #         metadata={},
    #         page_content="2024 WNBA Finals: New York Liberty defeated Minnesota Lynx 3-2 to win the championship.",
    #     ),
    #     Document(
    #         id="5556c3f1-b2bd-421d-bec0-014a58708b1e",
    #         metadata={},
    #         page_content="August 20, 2024: Las Vegas Aces 92, Phoenix Mercury 84. A'ja Wilson scored 35 points.",
    #     ),
    #     Document(
    #         id="1a92c07e-a632-4289-a1d1-cec8b9514001",
    #         metadata={},
    #         page_content="June 15, 2024: Indiana Fever 85, Chicago Sky 79. Caitlin Clark had 23 points and 8 assists.",
    #     ),
    #     Document(
    #         id="4807b9d0-cd58-4dd2-8bb8-989527ffefd6",
    #         metadata={},
    #         page_content="A'ja Wilson 2024 season stats: 26.9 PPG, 11.9 RPG, 2.6 BPG. Won MVP award.",
    #     ),
    #     Document(
    #         id="17de3537-e79e-4f74-baf1-c7a52d600f53",
    #         metadata={},
    #         page_content="Breanna Stewart 2024 stats: 20.4 PPG, 8.5 RPG, 3.5 APG.",
    #     ),
    # ]
    print("_______________retrieve_documents__________")
    print(docs)
    return {"documents": [doc.page_content for doc in docs]}


def call_agent(state: CustomState) -> dict:
    context = "\n\n".join(state["documents"])
    prompt = f"Context:\n{context}\n\nQuestion:{state['rewritten_question']}"
    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return {"answer": response["messages"][-1].content_blocks}


workflow = (
    StateGraph(CustomState)
    .add_node("rewrite_query", rewrite_query)
    .add_node("retrieve_documents", retrieve_documents)
    .add_node("call_agent", call_agent)
    .add_edge(START, "rewrite_query")
    .add_edge("rewrite_query", "retrieve_documents")
    .add_edge("retrieve_documents", "call_agent")
    .add_edge("call_agent", END)
    .compile()
)

result = workflow.invoke({"question": "Who won the 2024 WNBA Championship?"})
print("_______________final_answer__________")
print(result["answer"])
