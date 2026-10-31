
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# 1. Connect to Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# 2. Create a short-term memory checkpointer
memory = InMemorySaver()

# 3. Create an agent with memory enabled
agent = create_agent(
    model=llm,
    tools=[],
    checkpointer=memory,
    system_prompt=(
        "You are a helpful HVAC learning assistant. "
        "Use the conversation history to answer questions "
        "about information the user previously shared. "
        "Do not invent memories."
    ),
)

# Both interactions use the same conversation ID
config = {
    "configurable": {
        "thread_id": "hvac-learning-session-1"
    }
}

# 4. First interaction: share information
print("\n--- FIRST INTERACTION ---")

result1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Remember this for our conversation: "
                    "I am learning HVAC VRF/VRV system design. "
                    "I prefer beginner-friendly explanations "
                    "with practical examples."
                ),
            }
        ]
    },
    config=config,
)

print(result1["messages"][-1].content)

# 5. Second interaction: test the memory
print("\n--- SECOND INTERACTION ---")

result2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "What HVAC topic am I learning, and "
                    "how do I prefer explanations?"
                ),
            }
        ]
    },
    config=config,
)

print(result2["messages"][-1].content)
