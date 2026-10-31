
import json
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Shared LLM used by all three agents
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=600,
)


def ask_json(agent_name, instructions, data):
    """Call an agent and convert its response into a Python dictionary."""

    prompt = f"""
You are {agent_name} in an HVAC engineering workflow.

Instructions:
{instructions}

Input data:
{json.dumps(data, ensure_ascii=False)}

Return exactly one valid JSON object.
Do not use Markdown or code fences.
Do not invent missing engineering data.
Keep the response concise.
"""

    response = llm.invoke(prompt)
    content = response.content

    # Handle either text or a list of text blocks
    if not isinstance(content, str):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    # Extract the JSON object if the model adds extra text
    match = re.search(r"\{.*\}", content, re.DOTALL)

    if not match:
        raise ValueError(
            f"{agent_name} did not return a JSON object:\n{content}"
        )

    return json.loads(match.group(0))


def run_hvac_workflow(user_question):
    # AGENT 1: Extract the HVAC inputs
    hvac_expert = ask_json(
        "HVAC Expert",
        """
Extract information explicitly provided by the user.

Return these keys:
room_dimensions, floor_area, ceiling_height, window_area,
occupancy, lighting_and_equipment, ventilation,
indoor_target, outdoor_design_conditions, envelope_details,
missing_inputs.

Use "Not provided" for unknown individual values.
Use a list of missing input names for missing_inputs.
Do not estimate cooling capacity.
""",
        {"user_question": user_question},
    )

    # AGENT 2: Assess whether the data supports an estimate
    engineering_analyst = ask_json(
        "Engineering Analyst",
        """
Review the extracted HVAC inputs.

Return these keys:
calculation_status, missing_inputs, calculation_method,
cooling_capacity_kw, explanation.

Set calculation_status to "insufficient_data" if essential
room-load inputs are missing.

Set cooling_capacity_kw to null unless enough reliable
information is available for a defensible estimate.
Do not invent values or use a generic rule of thumb as
a substitute for a proper heat-load calculation.
""",
        {"hvac_expert_output": hvac_expert},
    )

    # AGENT 3: Review the preceding agents
    reviewer = ask_json(
        "HVAC Reviewer",
        """
Review both previous agents for completeness and consistency.

Return these keys:
review_status, issues, final_answer, safety_note.

Check whether missing inputs are clearly identified.
Do not approve unsupported cooling-capacity figures.
If data is insufficient, tell the user what to provide next.
A qualified HVAC engineer should validate final equipment
selection against manufacturer performance data.
""",
        {
            "hvac_expert_output": hvac_expert,
            "engineering_analyst_output": engineering_analyst,
        },
    )

    return {
        "hvac_expert": hvac_expert,
        "engineering_analyst": engineering_analyst,
        "reviewer": reviewer,
    }


if __name__ == "__main__":
    question = input(
        "\nEnter your HVAC / VRF question: "
    ).strip()

    if not question:
        print("Please enter a question.")
    else:
        try:
            result = run_hvac_workflow(question)

            print("\n" + "=" * 50)
            print("STRUCTURED HVAC MULTI-AGENT REPORT")
            print("=" * 50)

            print(json.dumps(result, indent=2, ensure_ascii=False))

        except json.JSONDecodeError:
            print("Error: An agent returned invalid JSON.")
            print("Please run the script again.")
        except Exception as error:
            print(f"Workflow error: {error}")
