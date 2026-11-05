
def calculate_wall_conduction(room_data):
    """
    Calculate sensible cooling load through externally
    exposed walls using the simplified U-A-DeltaT method.

    Required inputs:
    - external_wall_area_m2
    - wall_u_value_w_m2k
    - indoor_target_c
    - outdoor_design_c
    """

    required_fields = [
        "external_wall_area_m2",
        "wall_u_value_w_m2k",
        "indoor_target_c",
        "outdoor_design_c",
    ]

    missing_inputs = [
        field
        for field in required_fields
        if room_data.get(field) is None
    ]

    if missing_inputs:
        return {
            "status": "insufficient_data",
            "component": "wall_conduction",
            "cooling_load_w": None,
            "cooling_load_kw": None,
            "missing_inputs": missing_inputs,
            "explanation": (
                "External wall area, U-value and "
                "design temperatures are required."
            ),
        }

    area = room_data["external_wall_area_m2"]
    u_value = room_data["wall_u_value_w_m2k"]
    indoor_temp = room_data["indoor_target_c"]
    outdoor_temp = room_data["outdoor_design_c"]

    if area < 0 or u_value <= 0:
        raise ValueError("Invalid wall area or U-value.")

    delta_t = outdoor_temp - indoor_temp

    # Cooling load only; negative temperature difference
    # does not represent a positive cooling gain.
    cooling_load_w = max(
        0,
        u_value * area * delta_t
    )

    return {
        "status": "calculated",
        "component": "wall_conduction",
        "external_wall_area_m2": area,
        "wall_u_value_w_m2k": u_value,
        "temperature_difference_k": delta_t,
        "cooling_load_w": round(cooling_load_w, 2),
        "cooling_load_kw": round(cooling_load_w / 1000, 4),
        "missing_inputs": [],
        "explanation": (
            "Calculated using U × A × temperature difference."
        ),
    }