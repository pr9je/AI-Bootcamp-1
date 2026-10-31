
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

# 1. Connect to Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

# 2. Tool: Calculate floor area
@tool
def calculate_floor_area(length_m: float, width_m: float) -> float:
    """Calculate room floor area in square metres."""
    return length_m * width_m


# 3. Tool: Estimate cooling capacity
@tool
def calculate_cooling_capacity(
    length_m: float,
    width_m: float,
    watts_per_square_metre: float = 150.0,
) -> float:
    """Estimate cooling capacity in watts using room dimensions
    and an assumed watts-per-square-metre value."""
    return length_m * width_m * watts_per_square_metre


# 4. Tool: Convert watts to BTU per hour
@tool
def watts_to_btu_per_hour(watts: float) -> float:
    """Convert cooling capacity from watts to BTU per hour."""
    return watts * 3.412142


# 5. Create the agent
agent = create_agent(
    model=llm,
    tools=[
        calculate_floor_area,
        calculate_cooling_capacity,
        watts_to_btu_per_hour,
    ],
    system_prompt=(
        "You are an HVAC learning assistant. "
        "Use the available tools for calculations. "
        "Explain the calculations clearly. "
        "State that cooling capacity estimates are approximate "
        "and are not final VRF/VRV equipment selections."
    ),
)


# 6. Ask a question
question = (
    "For a room 5 metres long and 4 metres wide, "
    "calculate the floor area, estimate cooling capacity "
    "using 150 watts per square metre, and convert the result "
    "to BTU per hour. Use the tools."
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": question}
        ]
    }
)

# 7. Print the conversation and tool activity
for message in result["messages"]:
    print(f"\n--- {message.type.upper()} ---")

    if message.content:
        print(message.content)

    tool_calls = getattr(message, "tool_calls", [])
    if tool_calls:
        print("Requested tool calls:")
        for call in tool_calls:
            print(call)

    if message.type == "tool":
        print("Tool execution result:", message.content)
