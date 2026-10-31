
import json
import re
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1000,
)


# 1. Define the expected data structure
class HVACRoomData(BaseModel):
    room_dimensions: Optional[str] = None
    floor_area_m2: Optional[float] = Field(default=None, gt=0)
    ceiling_height_m: Optional[float] = Field(default=None, gt=0)
    window_area_m2: Optional[float] = Field(default=None, ge=0)
    occupancy: Optional[int] = Field(default=None, ge=0)
    lighting_load_w: Optional[float] = Field(default=None, ge=0)
    equipment_load_w: Optional[float] = Field(default=None, ge=0)
    indoor_target_c: Optional[float] = None
    outdoor_design_c: Optional[float] = None
    ventilation_rate_m3_s: Optional[float] = Field(default=None, ge=0)
    wall_u_value_w_m2k: Optional[float] = Field(default=None, gt=0)
    window_shgc: Optional[float] = Field(default=None, ge=0, le=1)
    roof_u_value_w_m2k: Optional[float] = Field(default=None, gt=0)
    infiltration_rate_m3_s: Optional[float] = Field(default=None, ge=0)
    occupant_sensible_heat_w: Optional[float] = Field(default=None, ge=0)
    roof_area_m2: Optional[float] = Field(default=None, ge=0)
    solar_gain_orientation: Optional[str] = None
    humidity_control_parameters: Optional[str] = None
    floor_location: Optional[str] = None




def ask_json(agent_name, instructions, data):
    prompt = f"""
    Role: {agent_name}

    Task:
    {instructions}

    Input:
    {json.dumps(data, ensure_ascii=False)}

    Return one complete, valid JSON object only.
    Use every requested key.
    Use null for unknown numeric values.
    Keep explanations under 40 words.
    Do not include Markdown.
    """


    response = llm.invoke(prompt)
    content = response.content

    # Handle text-block responses
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        content = "".join(parts)

    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            f"{agent_name} returned an empty response."
        )

    # First, try parsing the entire response
    try:
        result = json.loads(content.strip())
    except json.JSONDecodeError:
        # If extra text exists, try extracting the JSON object
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end <= start:
            print(f"\nRaw response from {agent_name}:")
            print(repr(content))
            raise ValueError(
                f"{agent_name} did not return a JSON object."
            )

        try:
            result = json.loads(content[start:end + 1])
        except json.JSONDecodeError as error:
            print(f"\nRaw response from {agent_name}:")
            print(content)
            raise ValueError(
                f"{agent_name} returned invalid JSON: {error}"
            ) from error

    if not isinstance(result, dict):
        raise ValueError(
            f"{agent_name} response must be a JSON object."
        )

    return result




def collect_missing_inputs(room):
    """Collect missing fields and validate the supplied values."""

    numeric_fields = {
        "ventilation_rate_m3_s": "Ventilation airflow (m³/s)",
        "wall_u_value_w_m2k": "Wall U-value (W/m²·K)",
        "window_shgc": "Window SHGC (0 to 1)",
        "roof_u_value_w_m2k": "Roof U-value (W/m²·K)",
        "infiltration_rate_m3_s": "Infiltration airflow (m³/s)",
        "occupant_sensible_heat_w": "Total occupant sensible heat (W)",
        "roof_area_m2": "Applicable roof area (m²)",
        "occupant_sensible_heat_w":"Total occupant sensible heat (W)",
    }

    text_fields = {
        "solar_gain_orientation": "Window orientation",
        "humidity_control_parameters": ("Indoor humidity target or design requirements"),
        "floor_location": ("Room floor location (top, intermediate, or ground)"),
    }

    data = room.model_dump()

    for field, description in {
        **numeric_fields,
        **text_fields,
    }.items():
        if data.get(field) is not None:
            continue

        while True:
            answer = input(
                f"Enter {description}, or type 'skip': "
            ).strip()

            if answer.lower() == "skip":
                break

            if not answer:
                print("Input cannot be empty. Try again or type 'skip'.")
                continue

            try:
                if field in numeric_fields:
                    value = float(answer)
                else:
                    value = answer

                    if field == "solar_gain_orientation":
                        allowed = {"north","south","east","west"}
                        if value.lower() not in allowed:
                            print("Enter north, south, east, west or skip.")
                            continue
                        value = value.lower()

                    elif field == "humidity_control_parameters":
                        if len(value) < 3:
                            print("Please enter a meaningful humidity requirement or skip")
                            continue


                                                                                                            

                candidate = {**data, field: value}
                validated_room = HVACRoomData.model_validate(
                    candidate
                )
                data = validated_room.model_dump()
                break

            except (ValueError, ValidationError) as error:
                print(f"Invalid input: {error}")

    return HVACRoomData.model_validate(data)





def check_required_inputs(data):
    """Return required fields that are missing or None."""

    required_fields = [
        "floor_area_m2",
        "ceiling_height_m",
        "window_area_m2",
        "occupancy",
        "lighting_load_w",
        "equipment_load_w",
        "indoor_target_c",
        "outdoor_design_c",
        "ventilation_rate_m3_s",
        "wall_u_value_w_m2k",
        "window_shgc",
        "roof_u_value_w_m2k",
        "infiltration_rate_m3_s",
        "occupant_sensible_heat_w",
        "roof_area_m2",
        "solar_gain_orientation",
        "humidity_control_parameters",
    ]

    return [
        field
        for field in required_fields
        if data.get(field) is None
    ]



def run_workflow(question):
    # AGENT 1: Extract room data
    extracted = ask_json(
        "HVAC Expert",
        """
Extract these fields:
room_dimensions, floor_area_m2, ceiling_height_m,
window_area_m2, occupancy, lighting_load_w,
equipment_load_w, indoor_target_c, outdoor_design_c,
ventilation_rate_m3_s, wall_u_value_w_m2k, window_shgc, roof_u_value_w_m2k,
infiltration_rate_m3_s,
occupant_sensible_heat_w,
roof_area_m2,
solar_gain_orientation,
humidity_control_parameters.

Use numeric values for numeric fields and null when unknown.
Calculate floor area only when room dimensions support it.
Calculate lighting and equipment loads from supplied ratings.
""",
        {"question": question},
    )

    # Pydantic validates the extracted fields
    try:
        room = HVACRoomData.model_validate(extracted)
    except ValidationError as error:
        print("\nData validation failed:")
        print(error)
        return

    validated_data = room.model_dump()

    missing_required = check_required_inputs(validated_data)

    if missing_required:
        print("\n--- COLLECTING MISSING INPUTS ---")
        room = collect_missing_inputs(room)
        validated_data = room.model_dump()

        missing_required = check_required_inputs(validated_data)

    if missing_required:
        print("\n--- CALCULATION READINESS ---")
        print("Status: INSUFFICIENT DATA")
        print("Missing required inputs:")

        for field in missing_required:
            print(f"- {field}")

        print("Cooling capacity has not been calculated.")
        return

    print("\nAll configured fields are populated.")




    

    # AGENT 2: Assess calculation readiness
    analysis = ask_json(
        "Engineering Analyst",
        
"""
Assess whether the supplied room data supports a reliable
cooling-load calculation.

Return exactly these keys:
calculation_status, missing_inputs,
cooling_capacity_kw, explanation.

Use "insufficient_data" if a defensible calculation
cannot be completed.
Set cooling_capacity_kw to null if insufficient data.
Use a JSON array for missing_inputs.
Do not invent values or provide an unsupported capacity.
"""
,
        {"validated_room_data": validated_data},
    )

    # AGENT 3: Review the assessment
    review = ask_json(
        "HVAC Reviewer",
        """
Review the room data and analyst assessment.

Return:
review_status, issues, final_answer, safety_note.

Check that insufficient data does not lead to an unsupported
cooling capacity. Mention that final equipment selection
requires manufacturer performance data and qualified review.
""",
        {
            "validated_room_data": validated_data,
            "engineering_analysis": analysis,
        },
    )

    print("\n--- ENGINEERING ANALYST ---")
    print(json.dumps(analysis, indent=2))

    print("\n--- REVIEWER ---")
    print(json.dumps(review, indent=2))


if __name__ == "__main__":
    question = input("Enter your HVAC / VRF question: ").strip()

    if not question:
        print("Please enter a question.")
    else:
        try:
            run_workflow(question)
        except (json.JSONDecodeError, ValueError) as error:
            print(f"\nWorkflow error: {error}")
        except Exception as error:
            print(f"\nUnexpected error: {error}")
