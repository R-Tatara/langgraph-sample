import logging
from typing import Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

logging.getLogger("google_genai.models").setLevel(logging.ERROR)
MODEL_NAME = "gemini-3.1-flash-lite"


class GraphState(TypedDict):
    input_text: str
    is_supported: bool
    output_text: str


class ClassificationResult(BaseModel):
    is_supported: Literal["YES", "NO"] = Field(
        description=(
            "YES if the chatbot can mechanically answer using only general "
            "knowledge or common sense, otherwise NO"
        )
    )


def build_graph():
    # Build and compile the state graph
    graph_builder = StateGraph(GraphState)

    # Register nodes
    graph_builder.add_node("classify", classify_node)
    graph_builder.add_node("answer", answer_node)
    graph_builder.add_node("fallback", fallback_node)

    # Set the entry point
    graph_builder.set_entry_point("classify")

    # Wire edges between nodes
    graph_builder.add_conditional_edges("classify", route_by_support)
    graph_builder.add_edge("answer", END)
    graph_builder.add_edge("fallback", END)
    return graph_builder.compile()


def classify_node(state: GraphState) -> GraphState:
    # Determine whether the input is supported
    is_supported = classify_input(state["input_text"])
    return {"is_supported": is_supported}


def answer_node(state: GraphState) -> GraphState:
    # Generate the primary output
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
    result = llm.invoke(state["input_text"])
    return {"output_text": extract_text_content(result.content)}


def fallback_node(state: GraphState) -> GraphState:
    # Return a fixed message
    return {"output_text": "I'm sorry, I cannot answer that question."}


def route_by_support(state: GraphState) -> Literal["answer", "fallback"]:
    # Decide the next node based on the classification result
    if state["is_supported"]:
        ret = "answer"
    else:
        ret = "fallback"
    return ret


def classify_input(input_text: str) -> bool:
    # Judge whether the chatbot can mechanically answer using general knowledge only
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
    structured_llm = llm.with_structured_output(ClassificationResult)
    prompt = (
        "You are a classifier for a chatbot. Determine whether the following "
        "question can be mechanically answered by the chatbot using only "
        "general knowledge or common sense, without needing access to "
        "specific account, contract, or internal system information.\n\n"
        f"Question: {input_text}"
    )
    result: ClassificationResult = structured_llm.invoke(prompt)
    return result.is_supported == "YES"


def extract_text_content(content: str | list) -> str:
    # Normalize LLM message content (str or list of content blocks) into plain text
    if isinstance(content, str):
        return content
    text_parts = [
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(text_parts)


def main():
    graph = build_graph()
    result = graph.invoke({"input_text": "What is the capital of France?"})
    print(result["output_text"])


if __name__ == "__main__":
    main()
