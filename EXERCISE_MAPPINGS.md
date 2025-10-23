# Exercise Name Mappings - Two-Level System

## Overview

Garmin FIT files use a **two-level hierarchical system** for exercise naming:

1. **Category Level** - High-level exercise categories (53 categories)
2. **Subtype Level** - Specific exercise variations within each category (1,846+ variations)

The strength training analyzer extracts the most specific name available:
- If `category_subtype` is provided → Use **specific exercise name** (e.g., "Barbell Power Clean")
- If only `category` is available → Use **category name** (e.g., "Olympic Lift")

## Two-Level Naming Examples

| Category | Subtype | Category Name | Specific Exercise Name |
|----------|---------|---------------|------------------------|
| 18 | None | **Olympic Lift** | (fallback to category) |
| 18 | 0 | Olympic Lift | **Barbell Hang Power Clean** |
| 18 | 2 | Olympic Lift | **Barbell Power Clean** |
| 18 | 5 | Olympic Lift | **Clean And Jerk** |
| 13 | None | **Hyperextension** | (fallback to category) |
| 13 | 0 | Hyperextension | **Back Extension With Opposite Arm And Leg Reach** |
| 13 | 33 | Hyperextension | **Swiss Ball Hyperextension** |
| 5 | 41 | Core | **Ghd Back Extensions** |
| 5 | 42 | Core | **Weighted Ghd Back Extensions** |
| 0 | None | **Bench Press** | (fallback to category) |
| 0 | 1 | Bench Press | **Barbell Bench Press** |
| 0 | 4 | Bench Press | **Close Grip Barbell Bench Press** |
| 0 | 8 | Bench Press | **Incline Barbell Bench Press** |

## Source Data

- **Package**: `garmin-fit-sdk` version 21.178.0
- **Profile Version**: 21.178.0
- **Last Updated**: October 23, 2025
- **Official Documentation**: https://developer.garmin.com/fit

## Exercise Categories (Level 1)

The library recognizes **53 high-level categories**:

### Strength Training (Traditional)
- Bench Press (0)
- Squat (28)
- Deadlift (8)
- Lunge (17)
- Olympic Lift (18)

### Upper Body
- Pull Up (21)
- Push Up (22)
- Row (23)
- Shoulder Press (24)
- Triceps Extension (30)
- Curl (7)
- Shrug (26)
- Lateral Raise (14)
- Flye (9)

### Lower Body
- Calf Raise (1)
- Leg Curl (15)
- Leg Raise (16)
- Hip Raise (10)
- Hip Stability (11)
- Hip Swing (12)
- Hyperextension (13)

### Core
- Sit Up (27)
- Crunch (6)
- Plank (19)
- Core (5)

### Cardio & Conditioning
- Cardio (2)
- Run (32)
- Run Indoor (52)
- Bike (33)
- Bike Outdoor (53)
- Indoor Bike (41)
- Elliptical (39)
- Stair Stepper (47)

### Functional & Alternative
- Carry (3)
- Chop (4)
- Total Body (29)
- Plyo (20)
- Warm Up (31)
- Banded Exercises (37)
- Battle Rope (38)
- Floor Climb (40)
- Indoor Row (42)
- Ladder (43)
- Sandbag (44)
- Sled (45)
- Sledge Hammer (46)
- Suspension (49)
- Tire (50)

### Specialized
- Move (35) - Mobility exercises
- Pose (36) - Yoga/flexibility poses
- Cardio Sensors (34)
- Shoulder Stability (25)

### Unknown
- Unknown (65534) - Garmin's unknown/unset category value

## Updating the Mappings

If you need to update the exercise mappings to match a newer version of the Garmin FIT SDK:

1. Install the latest Garmin FIT SDK:
   ```bash
   pip install --upgrade garmin-fit-sdk
   ```

2. Run the generation script:
   ```bash
   python generate_exercise_mapping.py
   ```

3. Copy the output to `src/fitanalyzer/constants.py`, replacing the existing `EXERCISE_CATEGORY_MAPPING` dictionary.

4. Run tests to ensure compatibility:
   ```bash
   make test
   make lint
   ```

## How Exercise Names are Extracted

The FIT file stores exercise categories as numeric codes in the `category` field of strength training sets. Our parser:

1. Reads the raw category value (can be a tuple like `(27, 28, 30)` or single value like `23`)
2. Extracts the first non-null, non-65534 value from tuples
3. Looks up the category ID in `EXERCISE_CATEGORY_MAPPING`
4. Returns the human-readable name (e.g., "Sit Up", "Row", "Bench Press")

## Example Usage

```python
from fitanalyzer.constants import EXERCISE_CATEGORY_MAPPING

# Get exercise name from category code
category_code = 28
exercise_name = EXERCISE_CATEGORY_MAPPING.get(category_code, "Unknown")
print(exercise_name)  # Output: "Squat"

# Handle tuple categories (take first valid value)
category_tuple = (27, 28, 30)
for cat_val in category_tuple:
    if cat_val is not None and cat_val != 65534:
        exercise_name = EXERCISE_CATEGORY_MAPPING.get(cat_val, f"Exercise {cat_val}")
        break
print(exercise_name)  # Output: "Sit Up"
```

## CSV Output Example

The `strength_training_summary.csv` file includes the `exercise_name` column:

```csv
activity_id,file,date,sport,sub_sport,set_number,set_type,exercise_name,category,category_subtype,repetitions,weight,duration,timestamp
20474406937,20474406937_ACTIVITY.fit,2025-09-23,training,strength_training,2,active,Sit Up,"(27, 28, 30)","(None, None, None)",10.0,0.0,42.221,2025-09-23 10:08:29+00:00
20474406937,20474406937_ACTIVITY.fit,2025-09-23,training,strength_training,10,active,Triceps Extension,"(30, 26, 26)","(None, 24, None)",10.0,14.0,26.207,2025-09-23 10:08:29+00:00
20474406937,20474406937_ACTIVITY.fit,2025-09-23,training,strength_training,18,active,Lunge,"(17, 20, 21)","(None, 33, None)",10.0,0.0,42.592,2025-09-23 10:08:29+00:00
```

## Notes

- The category mapping includes only high-level categories (e.g., "Bench Press" not "Close Grip Bench Press")
- For more specific exercise variations, check the Garmin FIT SDK Profile types like `bench_press_exercise_name`, `squat_exercise_name`, etc.
- Some categories may not be commonly used in typical strength training workouts
- Category 65534 is Garmin's special value for unknown/unrecognized exercises
