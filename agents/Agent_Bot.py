from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama          # ✅ Fixed import
from langchain_core.messages import HumanMessage

class AgentState(TypedDict):
    messages: list                                # ✅ Should be a list
    response: str

# Point to your local Ollama model
llm = ChatOllama(model="gemma:2b")               # ✅ Use available model

def process_state(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])      # LLM expects list of messages
    state["response"] = response.content
    return state

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("process", process_state)
graph.add_edge(START, "process")
graph.add_edge("process", END)

app = graph.compile()

# Run
query = input("Enter your query: ")
while(query.lower() != "exit" and query.lower() != "/bye"):
    result = app.invoke({"messages": [HumanMessage(content=query)]})  # ✅ Wrap in HumanMessage
    print("\nResponse:", result["response"])
    query = input("Enter your query (or 'exit' to quit): ")
