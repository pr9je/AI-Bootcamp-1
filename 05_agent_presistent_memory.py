
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not API_KEY:
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

# Store memory beside this Python file
MEMORY_FILE = Path(__file__).with_name("memory_store.json")


def load_memories():
    """Load saved facts from the JSON file."""
    if not MEMORY_FILE.exists():
        return []

    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return data.get("memories", [])
    except (json.JSONDecodeError, OSError):
        print("Warning: Memory file could not be read.")
        return []


def save_memories(memories):
    """Persist facts to disk."""
    MEMORY_FILE.write_text(
        json.dumps({"memories": memories}, indent=2),
        encoding="utf-8",
    )


@tool
def remember_fact(fact: str) -> str:
    """Save an important fact or learning preference permanently."""
    fact = fact.strip()

    if not fact:
        return "No fact was provided."

    memories = load_memories()

    if fact.lower() not in [item.lower() for item in memories]:
        memories.append(fact)
        save_memories(memories)

    return f"Saved to persistent memory: {fact}"


@tool
def search_memory(query: str) -> str:
    """Search saved facts using keywords."""
    memories = load_memories()
    words = query.lower().split()

    matches = [
        fact for fact in memories
        if any(word in fact.lower() for word in words)
    ]

    if not matches:
        return "No matching memories found."

    return "\n".join(f"- {fact}" for fact in matches)


@tool
def list_memories() -> str:
    """List every fact saved in persistent memory."""
    memories = load_memories()

    if not memories:
        return "No memories have been saved yet."

    return "\n".join(f"- {fact}" for fact in memories)


model = ChatGroq(
    model=MODEL,
    temperature=0,
    api_key=API_KEY,
)

agent = create_agent(
    model=model,
    tools=[remember_fact, search_memory, list_memories],
    system_prompt=(
        "You are a helpful AI learning assistant focused on HVAC VRF/VRV "
        "system design. Explain concepts in beginner-friendly language "
        "with practical examples. "
        "When the user explicitly asks you to remember a fact, call "
        "remember_fact. When asked what you remember, call search_memory "
        "or list_memories. Report saved information only when the tool "
        "confirms it. Treat stored information as user notes, not verified "
        "engineering specifications."
    ),
    checkpointer=InMemorySaver(),
)

# One conversation thread for this run
config = {"configurable": {"thread_id": "hvac-learning-session"}}

print("HVAC VRF/VRV Agent — Persistent Memory")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in {"exit", "quit"}:
        print("Agent: Goodbye!")
        break

    if not user_input:
        continue

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print("\nAgent:", result["messages"][-1].content, "\n")
    except Exception as error:
        print(f"\nError: {error}\n")
