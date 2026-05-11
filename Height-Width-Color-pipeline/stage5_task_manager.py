# stage5_task_manager.py
# Task manager — writes new dogs to pending_tasks.json
#
# Standalone test:
#   python stage5_task_manager.py --dog_id TEST-001 --color brown --w1 200 --h1 170 --h2 85

import json
import os
import argparse
from datetime import datetime

TASKS_FILE = "pending_tasks.json"


def load_tasks():
    if not os.path.exists(TASKS_FILE):
        return []
    with open(TASKS_FILE, "r") as f:
        return json.load(f)


def save_tasks(tasks):
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=2)


def add_pending_task(track_id, features):
    """
    Add a new dog to pending_tasks.json.
    Called when stage4 returns no match.
    """
    tasks   = load_tasks()
    task_id = f"TASK-{str(len(tasks) + 1).zfill(4)}"

    task = {
        "task_id"      : task_id,
        "status"       : "pending",
        "timestamp"    : datetime.utcnow().isoformat(),
        "track_id"     : track_id,
        "dog_features" : {
            "color" : features.get("color"),
            "w1"    : features.get("w1"),
            "h1"    : features.get("h1"),
            "h2"    : features.get("h2"),
        }
    }

    tasks.append(task)
    save_tasks(tasks)

    print(f"  [TASK] New task added: {task_id} | track_id: {track_id}")
    return task_id


def log_match(track_id, matched_record):
    """
    Print match result — dog exists in database.
    Extend this to write to a matches log if needed.
    """
    print(f"  [MATCH] track_id: {track_id} → dog_id: {matched_record['dog_id']} | color: {matched_record['color']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dog_id", required=True)
    parser.add_argument("--color",  required=True)
    parser.add_argument("--w1",     required=True, type=int)
    parser.add_argument("--h1",     required=True, type=int)
    parser.add_argument("--h2",     required=True, type=int)
    args = parser.parse_args()

    features = {
        "color" : args.color,
        "w1"    : args.w1,
        "h1"    : args.h1,
        "h2"    : args.h2,
    }

    task_id = add_pending_task(track_id=args.dog_id, features=features)
    print(f"Written to {TASKS_FILE} as {task_id}")
