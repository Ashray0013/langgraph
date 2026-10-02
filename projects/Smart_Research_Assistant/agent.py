from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from typing import Annotated, Sequence, TypedDict
from langgraph.graph import add_messages, StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from datetime import datetime
import re
import json
import os


# ============ MEMORY MANAGER ============
class KeywordMemory:
    """Stores only keywords and summaries, not full messages"""

    def __init__(self, storage_path="agent_memory.json", max_turns=10):
        self.storage_path = storage_path
        self.max_turns = max_turns
        self.memory = self._load_memory()

    def _load_memory(self):
        """Load memory from disk if exists"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"turns": [], "keyword_index": {}}
        return {"turns": [], "keyword_index": {}}

    def _save_memory(self):
        """Persist memory to disk"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def _extract_keywords(self, text: str) -> list:
        """
        Extract meaningful keywords from text using simple NLP rules.
        Filters out common stop words.
        """
        # Common English stop words to filter out
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
            'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
            'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
            'against', 'between', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
            'don', 'should', 'now', 'tell', 'explain', 'describe', 'please',
            'would', 'could', 'give', 'want', 'know', 'think', 'need'
        }

        # Clean and tokenize
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        words = text.split()

        # Filter stop words and short words
        keywords = [
            word for word in words
            if word not in stop_words and len(word) > 2
        ]

        # Get unique keywords, preserve order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        # Limit to top 8 most relevant keywords
        return unique_keywords[:8]

    def add_turn(self, user_query: str, assistant_response: str, tools_used: list = None):
        """Store a conversation turn with extracted keywords"""
        turn_id = f"turn_{len(self.memory['turns']) + 1}_{datetime.now().strftime('%H%M%S')}"

        # Extract keywords from BOTH query and response
        query_keywords = self._extract_keywords(user_query)
        response_keywords = self._extract_keywords(assistant_response)
        combined_keywords = list(set(query_keywords + response_keywords))

        # Create a summary (first 150 chars of response as preview)
        response_summary = assistant_response[:150].strip()
        if len(assistant_response) > 150:
            response_summary += "..."

        turn = {
            "id": turn_id,
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "query_keywords": query_keywords,
            "response_summary": response_summary,
            "all_keywords": combined_keywords,
            "tools_used": tools_used or []
        }

        self.memory["turns"].append(turn)

        # Update keyword index (keyword -> list of turn IDs)
        for kw in combined_keywords:
            if kw not in self.memory["keyword_index"]:
                self.memory["keyword_index"][kw] = []
            self.memory["keyword_index"][kw].append(turn_id)

        # Enforce max_turns limit (sliding window)
        if len(self.memory["turns"]) > self.max_turns:
            removed_turn = self.memory["turns"].pop(0)
            # Clean up keyword index
            for kw in removed_turn["all_keywords"]:
                if kw in self.memory["keyword_index"]:
                    if removed_turn["id"] in self.memory["keyword_index"][kw]:
                        self.memory["keyword_index"][kw].remove(removed_turn["id"])
                    if not self.memory["keyword_index"][kw]:
                        del self.memory["keyword_index"][kw]

        self._save_memory()

    def get_relevant_context(self, current_query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant past turns based on keyword overlap.
        Returns a formatted context string for the LLM.
        """
        current_keywords = set(self._extract_keywords(current_query))

        if not current_keywords or not self.memory["turns"]:
            return ""

        # Score each turn by keyword overlap
        scored_turns = []
        for turn in self.memory["turns"]:
            overlap = len(current_keywords & set(turn["all_keywords"]))
            if overlap > 0:
                scored_turns.append((overlap, turn))

        # Sort by relevance score
        scored_turns.sort(key=lambda x: x[0], reverse=True)
        top_turns = scored_turns[:top_k]

        if not top_turns:
            return ""

        # Format context
        context_lines = ["RELEVANT PAST CONVERSATIONS:"]
        for score, turn in top_turns:
            context_lines.append(
                f"- You previously answered about {', '.join(turn['query_keywords'][:3])}: "
                f"{turn['response_summary']}"
            )

        return "\n".join(context_lines)

    def get_full_context_for_agent(self, current_query: str) -> str:
        """Get formatted context string to inject into system prompt"""
        return self.get_relevant_context(current_query, top_k=2)

    def show_memory(self):
        """Display stored memory (for debugging)"""
        print("\n" + "=" * 60)
        print("📚 AGENT MEMORY (Keyword Storage)")
        print("=" * 60)
        for turn in self.memory["turns"]:
            print(f"\n🆔 {turn['id']} ({turn['timestamp']})")
            print(f"   Query: {turn['user_query']}")
            print(f"   Keywords: {turn['query_keywords']}")
            print(f"   Summary: {turn['response_summary']}")
            print(f"   Tools: {turn['tools_used']}")
        print("=" * 60 + "\n")

    def clear(self):
        """Clear all memory"""
        self.memory = {"turns": [], "keyword_index": {}}
        self._save_memory()
        print("🧹 Memory cleared!")


# ============ TOOLS ============
search_engine = DuckDuckGoSearchRun()
wiki_engine = WikipediaAPIWrapper()


@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo. Use this for current events, recent information, or general web queries.
"""
    return search_engine.run(query)


@tool
def wikipedia_tool(query: str) -> str:
    """Search Wikipedia for factual information. Use this for historical facts, definitions, and encyclopedia-style content."""
    return wiki_engine.run(query)


tools = [search_tool, wikipedia_tool]


# ============ MODEL ============
model = ChatOllama(model="llama3.2:3b").bind_tools(tools)


# ============ STATE ============
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ============ MEMORY INSTANCE ============
memory = KeywordMemory(max_turns=10)


# ============ NODES ============
def model_call(state: AgentState) -> AgentState:
    # Get last user query for memory retrieval
    last_user_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break

    # Retrieve relevant past context from memory
    past_context = memory.get_full_context_for_agent(last_user_msg or "")

    system_prompt = SystemMessage(content=(
        "You are a helpful research assistant. "
        "You have access to two tools:\n"
        "1. search_tool: Use for current events, recent news, and general web queries\n"
        "2. wikipedia_tool: Use for historical facts, definitions, and encyclopedia content\n"
        "Always use the appropriate tool to find accurate information before answering. "
        "After getting tool results, provide a clear, concise answer to the user.\n\n"
        "also cite the source of informations you use in your answer. "
        + (past_context if past_context else "No relevant past conversations found.")
    ))

    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


# ============ ROUTING ============
def should_continue(state: AgentState) -> str:
    """Decide whether to continue using tools or end"""
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"


# ============ GRAPH ============
graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "end": END},
)
graph.add_edge("tools", "agent")

app = graph.compile()


# ============ RUN ============
def run_agent():
    print("🔍 Research Assistant Ready! (type 'exit' to quit, 'memory' to view, 'clear' to reset)\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye! 👋")
                break

            if user_input.lower() == 'memory':
                memory.show_memory()
                continue

            if user_input.lower() == 'clear':
                memory.clear()
                continue

            if not user_input:
                continue

            # Run agent
            result = app.invoke({"messages": [HumanMessage(content=user_input)]})

            # Extract final answer
            final_response = result["messages"][-1].content

            # Track which tools were used
            tools_used = []
            for msg in result["messages"]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc["name"] not in tools_used:
                            tools_used.append(tc["name"])

            # Store in memory (KEYWORDS ONLY, not full messages)
            memory.add_turn(
                user_query=user_input,
                assistant_response=final_response,
                tools_used=tools_used
            )

            # Print final answer
            print("\n🤖 Assistant:", final_response)
            print(f"   [Tools used: {', '.join(tools_used) if tools_used else 'none'}]")
            print()

        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            print(f"\n⚠️ Error: {e}\n")


if __name__ == "__main__":
    run_agent()
