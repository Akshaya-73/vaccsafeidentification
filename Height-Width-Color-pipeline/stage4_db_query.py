# stage4_db_query.py
# Database query — checks if dog exists in dummy_db.json
# Replace load_db() with PostgreSQL query when ready.
#
# Standalone test:
#   python stage4_db_query.py --color brown --w1 210 --h1 180 --h2 90

import json
import os
import argparse

DUMMY_DB_PATH = "dummy_db.json"

# Tolerances — wider values compensate for average camera quality
# and varying distances between camera and dog
COLOR_MUST_MATCH = True   # color is exact match only
W1_TOLERANCE     = 40     # +/- pixels
H1_TOLERANCE     = 40     # +/- pixels
H2_TOLERANCE     = 25     # +/- pixels


def load_db():
    """
    Load dummy database from JSON.
    Replace this function with a PostgreSQL query when ready:

    SELECT dog_id, color, w1, h1, h2
    FROM dog_biometric
    JOIN dog ON dog.id = dog_biometric.dog_id
    JOIN sighting ON sighting.dog_id = dog.id
    WHERE sighting.ward_id = :ward_id
    """
    if not os.path.exists(DUMMY_DB_PATH):
        return []
    with open(DUMMY_DB_PATH, "r") as f:
        return json.load(f)


def query_db(features):
    """
    Match query features against all records in the database.

    Matching rules:
      1. Color must match exactly
      2. w1, h1, h2 must be within tolerance

    Returns matched record dict or None.
    """
    records = load_db()

    query_color = features.get("color", "unknown")
    query_w1    = features.get("w1")
    query_h1    = features.get("h1")
    query_h2    = features.get("h2")

    if query_color == "unknown" or None in (query_w1, query_h1, query_h2):
        return None

    for record in records:
        # color must match
        if COLOR_MUST_MATCH and record["color"] != query_color:
            continue

        # dimension tolerance check
        if abs(record["w1"] - query_w1) > W1_TOLERANCE:
            continue
        if abs(record["h1"] - query_h1) > H1_TOLERANCE:
            continue
        if abs(record["h2"] - query_h2) > H2_TOLERANCE:
            continue

        # all checks passed
        return record

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--color", required=True)
    parser.add_argument("--w1",    required=True, type=int)
    parser.add_argument("--h1",    required=True, type=int)
    parser.add_argument("--h2",    required=True, type=int)
    args = parser.parse_args()

    features = {
        "color" : args.color,
        "w1"    : args.w1,
        "h1"    : args.h1,
        "h2"    : args.h2,
    }

    match = query_db(features)

    if match:
        print(f"MATCH FOUND")
        print(f"  dog_id : {match['dog_id']}")
        print(f"  color  : {match['color']}")
        print(f"  w1     : {match['w1']}  h1: {match['h1']}  h2: {match['h2']}")
    else:
        print("NO MATCH — dog not in database")
