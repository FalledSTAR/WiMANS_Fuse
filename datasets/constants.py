ACTIVITY_TO_ID = {
    "nothing": 0,
    "walk": 1,
    "rotation": 2,
    "jump": 3,
    "wave": 4,
    "lie_down": 5,
    "pick_up": 6,
    "sit_down": 7,
    "stand_up": 8,
}

ID_TO_ACTIVITY = {idx: name for name, idx in ACTIVITY_TO_ID.items()}

ACTIVITY_COLS = [
    "user_1_activity",
    "user_2_activity",
    "user_3_activity",
    "user_4_activity",
    "user_5_activity",
    "user_6_activity",
]
