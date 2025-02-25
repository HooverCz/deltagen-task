import json
from typing import Annotated, TypedDict

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from src.llm import get_llm
from src.retriever import get_retriever


class State(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    retrieval_context: list[str]
    answer: str
    conversation: Annotated[list, add_messages]


def parse_docs(docs: list[bytes]) -> dict[str, list[dict[str, any]]]:
    """Splits base64-encoded images and texts.

    Args:
        docs (list[bytes]): A list of documents in bytes format.

    Returns:
        dict[str, list[dict[str, any]]]: A dictionary with keys 'images' and 'texts',
        each containing a list of parsed document data.
    """
    parsed_data: dict[str, list[dict[str, any]]] = {"images": [], "texts": []}

    for doc in docs:
        data: dict[str, any] = json.loads(doc.decode("utf-8"))
        key: str = "images" if data["type"] == "image" else "texts"
        parsed_data[key].append(data)

    return parsed_data


def retriever_node(state: State) -> dict[str, dict[str, list[dict[str, any]]]]:
    """Retrieves documents based on the question in the state.

    Args:
        state (State): The current state containing the question.

    Returns:
        dict[str, dict[str, list[dict[str, any]]]]: A dictionary with the key 'retrieval_context'
        containing cleaned documents.
    """
    question: str = state["question"]
    retriever: MultiVectorRetriever = get_retriever()
    docs: list[bytes] = retriever.invoke(question)
    cleaned_docs: dict[str, list[dict[str, any]]] = parse_docs(docs)
    print(cleaned_docs)

    return {
        "retrieval_context": cleaned_docs,
    }


def chat_node(state: State) -> dict[str, str]:
    """Generates an answer to the question using the retrieval context.

    Args:
        state (State): The current state containing the question and retrieval context.

    Returns:
        dict[str, str]: A dictionary with the key 'answer' containing the generated response.
    """
    question: str = state["question"]
    context: dict[str, list[dict[str, any]]] = state["retrieval_context"]
    print(context)

    context_text: str = ""
    if len(context["texts"]) > 0:
        for text_element in context["texts"]:
            context_text += text_element["content"] + "\n"

    # construct prompt with context (including images)
    prompt_template: str = f"""
    Answer the question based only on the following context, which can include text and image.

    Context: {context_text}
    Question: {question}
    """

    prompt_content: list[dict[str, any]] = [{"type": "text", "text": prompt_template}]

    if len(context["images"]) > 0:
        for image in context["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image['content']}"},
                }
            )

    # invoke LLM
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt_content)])
    return {
        "answer": response.content,
    }


def get_agents_graph() -> CompiledStateGraph:
    """Gets the graph representing the agents.

    The graph is a StateGraph where each node is one of the agents or tools.
    The edges represent the flow of the conversation.

    Returns:
        CompiledStateGraph: The compiled state graph representing the agents.
    """
    workflow: StateGraph = StateGraph(State)

    # Add nodes
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("general_chat", chat_node)

    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "general_chat")
    workflow.add_edge("general_chat", END)

    # Compile
    memory = MemorySaver()
    graph: CompiledStateGraph = workflow.compile(checkpointer=memory)
    return graph
