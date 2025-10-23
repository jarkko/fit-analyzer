#!/usr/bin/env python3
"""Generate comprehensive exercise mappings from Garmin FIT SDK Profile

This script generates TWO mappings:
1. EXERCISE_CATEGORY_MAPPING - High-level categories (e.g., "Olympic Lift", "Hyperextension")
2. EXERCISE_SUBTYPE_MAPPING - Specific exercises within each category (e.g., "Barbell Power Clean")
"""

from garmin_fit_sdk import Profile


def snake_to_title(snake_str):
    """Convert snake_case to Title Case"""
    return " ".join(word.capitalize() for word in snake_str.split("_"))


# Get the exercise category mapping from Garmin SDK
exercise_categories = Profile["types"]["exercise_category"]

# Convert to the format we need
category_mapping = {}
for category_id, category_name in exercise_categories.items():
    cat_id = int(category_id)
    display_name = snake_to_title(category_name)
    category_mapping[cat_id] = display_name

# Sort by key
sorted_category_mapping = dict(sorted(category_mapping.items()))

# Generate the category mapping code
print("# Auto-generated from Garmin FIT SDK Profile")
print("# SDK Version: 21.178.0")
print("# Generated on: 2025-10-23")
print()
print("# High-level exercise categories")
print("EXERCISE_CATEGORY_MAPPING = {")
for cat_id, name in sorted_category_mapping.items():
    print(f"    {cat_id}: \"{name}\",")
print("}")
print()
print(f"# Total exercise categories: {len(category_mapping)}")
print()
print()

# Now generate the subtype mapping for specific exercises
# This maps (category_id, subtype_id) -> specific_exercise_name
print("# Specific exercise names within each category")
print("# Format: (category_id, subtype_id) -> exercise_name")
print("EXERCISE_SUBTYPE_MAPPING = {")

# Map category names to their exercise_name type
category_to_type_mapping = {
    "bench_press": "bench_press_exercise_name",
    "calf_raise": "calf_raise_exercise_name",
    "cardio": "cardio_exercise_name",
    "carry": "carry_exercise_name",
    "chop": "chop_exercise_name",
    "core": "core_exercise_name",
    "crunch": "crunch_exercise_name",
    "curl": "curl_exercise_name",
    "deadlift": "deadlift_exercise_name",
    "flye": "flye_exercise_name",
    "hip_raise": "hip_raise_exercise_name",
    "hip_stability": "hip_stability_exercise_name",
    "hip_swing": "hip_swing_exercise_name",
    "hyperextension": "hyperextension_exercise_name",
    "lateral_raise": "lateral_raise_exercise_name",
    "leg_curl": "leg_curl_exercise_name",
    "leg_raise": "leg_raise_exercise_name",
    "lunge": "lunge_exercise_name",
    "olympic_lift": "olympic_lift_exercise_name",
    "plank": "plank_exercise_name",
    "plyo": "plyo_exercise_name",
    "pull_up": "pull_up_exercise_name",
    "push_up": "push_up_exercise_name",
    "row": "row_exercise_name",
    "shoulder_press": "shoulder_press_exercise_name",
    "shoulder_stability": "shoulder_stability_exercise_name",
    "shrug": "shrug_exercise_name",
    "sit_up": "sit_up_exercise_name",
    "squat": "squat_exercise_name",
    "total_body": "total_body_exercise_name",
    "triceps_extension": "triceps_extension_exercise_name",
    "warm_up": "warm_up_exercise_name",
    "run": "run_exercise_name",
    "bike": "bike_exercise_name",
    "banded_exercises": "banded_exercises_exercise_name",
    "battle_rope": "battle_rope_exercise_name",
    "elliptical": "elliptical_exercise_name",
    "floor_climb": "floor_climb_exercise_name",
    "indoor_bike": "indoor_bike_exercise_name",
    "indoor_row": "indoor_row_exercise_name",
    "ladder": "ladder_exercise_name",
    "sandbag": "sandbag_exercise_name",
    "sled": "sled_exercise_name",
    "sledge_hammer": "sledge_hammer_exercise_name",
    "stair_stepper": "stair_stepper_exercise_name",
    "suspension": "suspension_exercise_name",
    "tire": "tire_exercise_name",
    "run_indoor": "run_indoor_exercise_name",
    "bike_outdoor": "bike_outdoor_exercise_name",
    "move": "move_exercise_name",
    "pose": "pose_exercise_name",
}

total_subtypes = 0
for category_id, category_name in sorted(exercise_categories.items(), key=lambda x: int(x[0]) if x[0] != '65534' else 99999):
    if category_name == "unknown" or category_name == "cardio_sensors":
        continue

    cat_id = int(category_id)
    type_name = category_to_type_mapping.get(category_name)

    if type_name and type_name in Profile["types"]:
        subtypes = Profile["types"][type_name]
        for subtype_id, subtype_name in sorted(subtypes.items(), key=lambda x: int(x[0]) if x[0] != '65534' else 99999):
            if subtype_name != "unknown":
                sub_id = int(subtype_id)
                display_name = snake_to_title(subtype_name)
                print(f"    ({cat_id}, {sub_id}): \"{display_name}\",")
                total_subtypes += 1

print("}")
print()
print(f"# Total specific exercise subtypes: {total_subtypes}")
