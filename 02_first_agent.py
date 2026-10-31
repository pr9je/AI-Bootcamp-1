
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

# 1. Connect to the language model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# 2. Define a tool the agent can use
@tool
def calculate_cooling_capacity(
    length_m: float,
    width_m: float,
    capacity_w_per_m2: float = 150.0,
) -> float:
    """Estimate cooling capacity in watts using floor area
    multiplied by an assumed watts-per-square-metre value.
    This is a rough illustrative estimate, not HVAC design advice.
    """
    return length_m * width_m * capacity_w_per_m2

# 3. Create the agent and give it access to the tool
agent = create_agent(
    model=llm,
    tools=[calculate_cooling_capacity],
    system_prompt=(
        "You are an HVAC learning assistant. "
        "Use the cooling capacity tool when a user asks for "
        "an estimate based on room dimensions. "
        "Explain that the result is only a rough estimate."
    ),
)


# 4. Ask the agent a question and inspect its messages
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Estimate cooling capacity for a room "
                    "that is 5 metres long and 4 metres wide."
                ),
            }
        ]
    }
)

# 5. Print every message to observe the agent workflow
for message in response["messages"]:
    print(f"\n--- {message.__class__.__name__} ---")
    print(message.content)

    # Show tool calls requested by the model, if any
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        print("Tool calls:", tool_calls)

    # Show tool execution results, if any
    if getattr(message, "type", "") == "tool":
        print("Tool result:", message.content)
