
from typing import Optional
from pydantic import BaseModel, Field


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
    ventilation_rate_m3_s: Optional[float] = Field(
        default=None, ge=0
    )
    wall_u_value_w_m2k: Optional[float] = Field(
        default=None, gt=0
    )
    window_shgc: Optional[float] = Field(
        default=None, ge=0, le=1
    )


if __name__ == "__main__":
    sample_data = {
        "room_dimensions": "5m x 4m",
        "floor_area_m2": 20,
        "ceiling_height_m": 3,
        "window_area_m2": 2,
        "occupancy": 2,
        "lighting_load_w": 80,
        "equipment_load_w": 300,
        "indoor_target_c": 24,
        "outdoor_design_c": 40
    }

    room = HVACRoomData.model_validate(sample_data)

    print("Validated HVAC room data:")
    print(room.model_dump_json(indent=2))

    print("\nMissing fields:")
    for field, value in room.model_dump().items():
        if value is None:
            print(f"- {field}")
