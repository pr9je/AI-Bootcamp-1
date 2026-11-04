
import json
import re
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langchain_groq import ChatGroq
from hvac_load_calculator import calculate_wall_conduction_load

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1500,
)


# 1. Define the expected data structure

class HVACRoomData(BaseModel):
    room_dimensions: Optional[str] = None

    # Room geometry
    length_m: Optional[float] = Field(default=None, gt=0)
    width_m: Optional[float] = Field(default=None, gt=0)
    floor_area_m2: Optional[float] = Field(default=None, gt=0)
    ceiling_height_m: Optional[float] = Field(default=None, gt=0)

    # Envelope areas
    wall_area_m2: Optional[float] = Field(default=None, ge=0)
    window_area_m2: Optional[float] = Field(default=None, ge=0)

    # Internal loads
    occupancy: Optional[int] = Field(default=None, ge=0)
    lighting_load_w: Optional[float] = Field(default=None, ge=0)
    equipment_load_w: Optional[float] = Field(default=None, ge=0)
    occupant_sensible_heat_w: Optional[float] = Field(default=None, ge=0)

    # Indoor and outdoor conditions
    indoor_target_c: Optional[float] = None
    outdoor_design_c: Optional[float] = None
    humidity_control_parameters: Optional[str] = None

    # Ventilation and infiltration
    ventilation_rate_m3_s: Optional[float] = Field(default=None, ge=0)
    infiltration_rate_m3_s: Optional[float] = Field(default=None, ge=0)

    # Thermal properties
    wall_u_value_w_m2k: Optional[float] = Field(default=None, gt=0)
    window_shgc: Optional[float] = Field(default=None, ge=0, le=1)
    solar_gain_orientation: Optional[str] = None

    # Roof properties
    roof_u_value_w_m2k: Optional[float] = Field(default=None, gt=0)
    roof_area_m2: Optional[float] = Field(default=None, ge=0)

    # Room location
    floor_location: Optional[str] = None

    # Ventilation conditions
    ventilation_supply_air_temp_c: Optional[float] = None

    # Latent heat gains
    occupant_latent_heat_w: Optional[float] = Field(default=None,ge=0)

    latent_heat_load_w: Optional[float] = Field(default=None,ge=0)

    external_wall_area_m2: Optional[float] = Field(default=None,ge=0)

    




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
    print(f"\nAgent: {agent_name}")
    print("Finish reason:", response.response_metadata.get("finish_reason"))
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
    """Collect missing fields and validate supplied values."""

    numeric_fields = {
        "ventilation_rate_m3_s": "Ventilation airflow (m³/s)",
        "wall_u_value_w_m2k": "Wall U-value (W/m²·K)",
        "window_shgc": "Window SHGC (0 to 1)",
        "roof_u_value_w_m2k": "Roof U-value (W/m²·K)",
        "infiltration_rate_m3_s": "Infiltration airflow (m³/s)",
        "occupant_sensible_heat_w": "Total occupant sensible heat (W)",
        "roof_area_m2": "Applicable roof area (m²)",
        "external_wall_area_m2": "Externally exposed opaque wall area (m²)",
    }

    text_fields = {
        "solar_gain_orientation": "Window orientation",
        "humidity_control_parameters": (
            "Indoor humidity target or design requirements"
        ),
        "floor_location": (
            "Room floor location (top, intermediate, or ground)"
        ),
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

            # Handle skip before numeric conversion
            if answer.lower() in {"skip", "null", "none", "n/a"}:
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
                        allowed = {"north", "south", "east", "west"}

                        if value.lower() not in allowed:
                            print("Enter north, south, east, west, or skip.")
                            continue

                        value = value.lower()

                    elif field == "floor_location":
                        allowed = {"top", "intermediate", "ground"}

                        if value.lower() not in allowed:
                            print("Enter top, intermediate, ground, or skip.")
                            continue

                        value = value.lower()

                    elif field == "humidity_control_parameters":
                        if len(value) < 3:
                            print("Enter a meaningful humidity requirement.")
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
        "infiltration_rate_m3_s",
        "humidity_control_parameters",
        "floor_location",
    ]

    missing = [
        field
        for field in required_fields
        if data.get(field) is None
    ]

    if data.get("floor_location") == "top":
        for field in ["roof_u_value_w_m2k", "roof_area_m2"]:
            if data.get(field) is None:
                missing.append(field)

    return missing




def filter_irrelevant_roof_requirements(analysis, validated_data):
    """
    Remove roof requirements when the room is not
    directly exposed to the roof in this simplified model.
    """

    floor_location = validated_data.get("floor_location")

    roof_fields = {
        "roof_u_value_w_m2k",
        "roof_area_m2",
    }

    # Roof fields are applicable only to top-floor rooms
    if floor_location in {"intermediate", "ground"}:

        missing_inputs = analysis.get("missing_inputs", [])

        analysis["missing_inputs"] = [
            field for field in missing_inputs
            if field not in roof_fields
        ]

    # Update status if roof requirements were the only issue
    if not analysis.get("missing_inputs"):
        analysis["calculation_status"] = "ready_for_calculation"

    return analysis

def calculate_wall_area(room_data):
    length = room_data.get("length_m")
    width = room_data.get("width_m")
    height = room_data.get("ceiling_height_m")

    if length is None or width is None or height is None:
        return None

    gross_wall_area = 2 * (length + width) * height

    window_area = room_data.get("window_area_m2") or 0

    return max(gross_wall_area - window_area, 0)



def calculated_room_geometry(room):
    """
    Calculate floor area, room volume,
    gross wall area and opaque wall area.
    """

    # Check whether room dimensions are available
    if room.length_m is not None and room.width_m is not None:

        room.floor_area_m2 = (
            room.length_m * room.width_m
        )

    # Calculate room volume
    if (
        room.length_m is not None
        and room.width_m is not None
        and room.ceiling_height_m is not None
    ):

        room_volume_m3 = (
            room.length_m
            * room.width_m
            * room.ceiling_height_m
        )

        # Calculate gross wall area
        gross_wall_area = (
            2 * (room.length_m + room.width_m)
            * room.ceiling_height_m
        )

        # Subtract window area
        window_area = room.window_area_m2 or 0

        room.wall_area_m2 = max(
            gross_wall_area - window_area,
            0
        )

        print("\n--- ROOM GEOMETRY ---")
        print(f"Floor area: {room.floor_area_m2:.2f} m²")
        print(f"Room volume: {room_volume_m3:.2f} m³")
        print(f"Gross wall area: {gross_wall_area:.2f} m²")
        print(f"Opaque wall area: {room.wall_area_m2:.2f} m²")

    return room


def calculate_internal_loads(room):
    """
    Calculate lighting and equipment loads
    from their supplied ratings.
    """

    lighting_load = room.lighting_load_w
    equipment_load = room.equipment_load_w

    if lighting_load is None or equipment_load is None:
        return None

    total_internal_load = lighting_load + equipment_load

    print("\n--- INTERNAL LOAD CALCULATIONS ---")
    print(f"Lighting load: {lighting_load:.2f} W")
    print(f"Equipment load: {equipment_load:.2f} W")
    print(f"Total internal electrical load: {total_internal_load:.2f} W")

    return total_internal_load


def run_workflow(question):
    # AGENT 1: Extract room data
    
    extracted = ask_json(
        "HVAC Expert",
        """
    Extract the following fields from the user question:

    room_dimensions
    length_m
    width_m
    floor_area_m2
    ceiling_height_m
    wall_area_m2
    external_wall_area_m2
    window_area_m2
    occupancy
    lighting_load_w
    equipment_load_w
    indoor_target_c
    outdoor_design_c
    ventilation_rate_m3_s
    wall_u_value_w_m2k
    window_shgc
    roof_u_value_w_m2k
    infiltration_rate_m3_s
    occupant_sensible_heat_w
    roof_area_m2
    solar_gain_orientation
    humidity_control_parameters
    floor_location
    ventilation_supply_air_temp_c
    occupant_latent_heat_w
    latent_heat_load_w

    Rules:
    1. Return all 27 requested fields..
    2. Use numeric values for numeric fields.
    3. Use null for unknown values.
    4. Calculate floor area from dimensions when possible.
    5. Calculate lighting and equipment loads from supplied ratings.
    6. Extract west-facing orientation when stated.
    7. Never invent missing engineering data.
    8. Extract length_m and width_m separately when dimensions are provided.
    9. Never invent wall_area_m2.
    10. Return null for unavailable dimensions.
    11. Extract these values only when explicitly supplied.
    12. Return null when unknown.
    13. Never estimate or invent latent heat values.
    14. Do not treat occupant latent heat as the total room latent heat.
    15. Return exactly one JSON object.
    16. Include every requestd field.
    17. Use null for every unavailable field.
    18. Do not include Markdown or explanatory text.

    Return one compact JSON object only.
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

    #Calculate room gemoetry 
    room = calculated_room_geometry(room)

    calculate_internal_loads(room)

    validated_data = room.model_dump()

    missing_required = check_required_inputs(validated_data)

    if missing_required:
        print("\n--- COLLECTING MISSING INPUTS ---")
        room = collect_missing_inputs(room)

        room = calculated_room_geometry(room)

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
Assess the supplied HVAC room data.

Return:
calculation_status, missing_inputs,
cooling_capacity_kw, explanation.

Rules:
1. Respect the supplied floor_location.
2. If floor_location is "intermediate" or "ground",
   do not require roof_u_value_w_m2k or roof_area_m2.
3. If floor_location is "top", assess roof data.
4. Identify other missing engineering parameters.
5. Never invent missing values.
6. Do not calculate cooling capacity using assumptions
   that have not been explicitly provided.
7. Set cooling_capacity_kw to null if a reliable
   cooling-load calculation cannot be completed.
8. Return one valid JSON object.
""",
        {"validated_room_data": validated_data},
    )

    # PYTHON: Enforce conditional roof requirements
    analysis = filter_irrelevant_roof_requirements(
        analysis,
        validated_data
    )

    # AGENT 3: Review the assessment
    review = ask_json(
        "HVAC Reviewer",
        """
Review the supplied HVAC room data and engineering assessment.

Return:
review_status, issues, final_answer, safety_note.

Rules:
1. Respect floor_location in validated_room_data.
2. For intermediate and ground floors, do not report
   missing roof U-value or roof area as issues under
   this simplified workflow.
3. For top floors, review applicable roof requirements.
4. Check whether the analyst identified missing inputs.
5. Do not approve unsupported cooling capacity.
6. Manufacturer performance data and qualified
   engineering review are required for final VRF selection.
7. Return one valid JSON object.
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
