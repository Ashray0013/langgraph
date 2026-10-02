from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import add_messages, StateGraph, START, END
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together. Use this for addition queries."""
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers. Use this for subtraction queries."""
    return a - b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers. Use this for multiplication queries."""
    return a * b



tools = [add, subtract, multiply]

# ✅ Use a model that handles tool calling well
model = ChatOllama(model="llama3.2:3b").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=(
        "You are my AI assistant. "
        "You have access to math tools (add, subtract, multiply). "
        "Use them whenever the user asks a math question."
    ))
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    print(f"making decision\n\n{messages}")
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"


graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


user_input = input("Enter: ")

while user_input != "exit":
    inputs = {"messages": [("user", user_input)]}
    print_stream(app.stream(inputs, stream_mode="values"))
    user_input = input("Enter: ")
