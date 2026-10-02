from typing import TypedDict, List , Union
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama          # ✅ Fixed import
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]              # ✅ Should be a list
# Point to your local Ollama model
llm = ChatOllama(model="gemma:2b")               # ✅ Use available model

def process_state(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])      # LLM expects list of messages
    
    state["messages"].append(AIMessage(content=response.content))
    print(f"AI response: {response.content}")

    return state

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("process", process_state)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")


with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")