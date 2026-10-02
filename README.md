# LangGraph Learning Repository

This repository is my hands-on learning space for building stateful AI workflows with LangGraph, LangChain, and local Ollama models.

The examples start with very small graphs and gradually introduce branching, loops, chat history, tool calling, document editing, retrieval-augmented generation (RAG), and persistent memory. The goal is to understand how an AI application can be represented as a graph of state transitions instead of as one large function.

## What I Learned

### 1. A graph is a workflow over shared state

LangGraph applications usually have:

- A typed state object, often a `TypedDict`.
- Nodes that read state and return updates.
- Edges that determine what runs next.
- A compiled application that can be invoked with initial state.

The smallest examples in `basic/` show how to create a `StateGraph`, add a node, define the entry and finish points, compile the graph, and invoke it.

### 2. Typed state makes data flow visible

The basic exercises pass values such as messages, names, ages, numbers, operations, and results through a shared state. Using `TypedDict` makes the expected shape of that state explicit and makes each node easier to understand.

### 3. Nodes can be composed sequentially

`basic/ex3.ipynb` builds a result through several nodes. One node creates a greeting, another appends an age, and another appends skills. This demonstrates how a larger task can be divided into small, testable steps.

### 4. Conditional edges choose the next step

The arithmetic examples and `agents/Conditional_Agent.ipynb` route execution according to an operation. A graph can therefore behave differently for different inputs without putting all control flow inside one node.

### 5. Graphs can loop

`basic/loop.ipynb` demonstrates a graph that continues through conditional routing until a number is guessed or the attempt limit is reached. This is the foundation for agent loops where a model repeatedly reasons, calls a tool, receives the result, and decides whether more work is needed.

### 6. Messages are state too

The agent examples store `HumanMessage`, `AIMessage`, and `ToolMessage` objects in graph state. `add_messages` provides message-aware state updates, allowing a conversation to accumulate without manually rebuilding the complete history each time.

### 7. Tools connect the model to actions

Functions decorated with LangChain's `@tool` can be exposed to a chat model. The model decides when a tool is needed, LangGraph routes the tool call to `ToolNode` or a custom tool node, and the tool result is returned to the model for a final answer.

### 8. ReAct-style agents alternate between reasoning and action

`agents/ReAct.py` contains addition, subtraction, and multiplication tools. Its graph cycles between the model and the tools until the model produces a response without another tool call.

### 9. Memory can be designed for a specific purpose

`agents/Memory_Agent.py` keeps conversation messages in memory and writes a text log. The Smart Research Assistant uses a different approach: `KeywordMemory` stores recent queries, extracted keywords, short answer summaries, timestamps, and tool names in JSON. It then retrieves older turns when their keywords overlap with a new question.

This shows that “memory” does not have to mean storing every message forever. It can be a bounded, task-specific retrieval system.

### 10. RAG combines retrieval with generation

`agents/rag_agent.py` demonstrates the main RAG pipeline:

1. Load a PDF with `PyPDFLoader`.
2. Split the pages into overlapping chunks.
3. Create embeddings with Ollama's `nomic-embed-text` model.
4. Store the chunks in Chroma.
5. Retrieve the most similar chunks for a question.
6. Give the retrieved context to an Ollama chat model through a tool.

The model is not expected to know the entire document in advance; it can retrieve relevant source text before answering.

## Repository Structure

```text
.
├── basic/
│   ├── HalloWorld.ipynb       # First StateGraph example
│   ├── ex1.ipynb              # Basic state mutation
│   ├── ex2.ipynb              # Typed inputs and arithmetic
│   ├── ex3.ipynb              # Sequential nodes
│   ├── multiple_inputs.ipynb  # Lists and aggregation
│   ├── loop.ipynb             # Conditional routing and loops
│   └── logging.txt            # Example conversation output
├── agents/
│   ├── Agent_Bot.py           # Minimal Ollama chatbot
│   ├── Memory_Agent.py        # Conversation history and logging
│   ├── ReAct.py               # Tool-calling math agent
│   ├── Drafter.py             # Document editing and saving tools
│   ├── rag_agent.py           # PDF retrieval-augmented generation
│   ├── Conditional_Agent.ipynb
│   ├── Calendar-Spring-2026.pdf
│   └── chroma.sqlite3         # Generated Chroma data
└── projects/
    └── Smart_Research_Assistant/
        ├── agent.py           # Web research agent with memory
        └── agent_memory.json  # Persistent sample memory
```

The Chroma database files are generated data, not source code. They can be recreated by the RAG workflow.

## Requirements

- Python 3.10 or newer.
- Jupyter support for the notebooks.
- Ollama installed and running locally.
- Internet access for DuckDuckGo and Wikipedia searches.
- The PDF used by the RAG example.

Install the Python packages in the active virtual environment:

```powershell
python -m pip install langgraph langchain-core langchain-ollama langchain-community langchain-text-splitters langchain-chroma pypdf numpy duckduckgo-search wikipedia jupyter
```

Download the local models used by the examples:

```powershell
ollama pull gemma:2b
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

`gemma:2b` is used by the basic chatbot and memory agent. The tool-calling, RAG, and research examples use `llama3.2:3b`; the RAG example also uses `nomic-embed-text` for embeddings.

## Running the Examples

### Notebooks

Open the `.ipynb` files in VS Code or Jupyter and run the cells in order. A useful progression is:

1. `basic/HalloWorld.ipynb`
2. `basic/ex1.ipynb`
3. `basic/ex2.ipynb`
4. `basic/ex3.ipynb`
5. `basic/multiple_inputs.ipynb`
6. `basic/loop.ipynb`
7. `agents/Conditional_Agent.ipynb`

### Chatbot and memory examples

Run these from the repository root:

```powershell
python agents/Agent_Bot.py
python agents/Memory_Agent.py
```

Type `exit` or `/bye` when supported by the program. The memory example writes its log relative to the current working directory.

### Tool-calling and document drafting

```powershell
python agents/ReAct.py
python agents/Drafter.py
```

`ReAct.py` expects a tool-calling-capable Ollama model. `Drafter.py` can update and save text documents using its tools and the sample files in `agents/`.

### Smart Research Assistant

Run it from its own directory so `agent_memory.json` resolves correctly:

```powershell
Set-Location projects/Smart_Research_Assistant
python agent.py
```

Supported commands are `memory`, `clear`, `exit`, and `quit`. Normal questions can be answered through DuckDuckGo or Wikipedia, and relevant summaries from earlier turns are included when keyword overlap is found.

## RAG Example Status

The RAG example is a learning draft and needs these fixes before it runs reliably:

- Replace the undefined `our_tool` reference in the `tools_dict` comprehension with the actual tool variable.
- Resolve `Calendar-Spring-2026.pdf` relative to the script or run the script from the directory containing the PDF.
- Replace the machine-specific Chroma persistence path with a repository-relative path such as `agents/chroma_data`.
- Avoid rebuilding the vector store every time the module is imported; create it once and load it on later runs.
- Add a `__main__` guard so importing the module does not immediately start the interactive loop.

These are useful next steps because they turn the demonstration into a repeatable application.

## Important Design Lessons

- Keep graph state small and explicit.
- Give tools precise names, type hints, and descriptions so the model can select them correctly.
- Use conditional edges for routing decisions that belong to the workflow.
- Keep tool execution separate from model calls so each part can be tested independently.
- Bound persistent memory and store only what the application needs.
- Treat retrieved document text as evidence and ask the model to cite its sources.
- Use paths based on the project location rather than a single developer's absolute Windows path.
- Add `if __name__ == "__main__":` guards to scripts that should be importable.

## Security and Repository Hygiene

Do not commit API keys or other secrets. The workspace contains a `.env` file, so it should be checked for credentials, excluded from version control, and replaced with a `.env.example` containing placeholders if environment variables are needed. Rotate any key that may already have been exposed.

Generated vector-store files and runtime memory files should also be treated as local artifacts unless there is a deliberate reason to version them.

## Possible Next Steps

- Add a `requirements.txt` or `pyproject.toml` with pinned dependencies.
- Add tests for graph routing, tool functions, memory eviction, and keyword retrieval.
- Refactor shared agent state and message utilities into reusable modules.
- Fix and parameterize the RAG example.
- Add source links and metadata to retrieved chunks.
- Add structured logging and error handling around model and network calls.
- Replace keyword overlap with embeddings when the memory store grows.
- Add configuration for model names, paths, and maximum memory size.
