"""Find register-like numbers in a PaddleOCR result JSON.

A match is a mostly-numeric string whose nearest label (left of it or above it)
contains the word "register".
"""

import argparse
import json
import re
import sys

KEYWORD = "register"
MIN_LEN = 5
MIN_DIGIT_RATIO = 0.6


def digit_ratio(text):
    stripped = re.sub(r"[\s./\-]", "", text)
    if not stripped:
        return 0.0
    return sum(c.isdigit() for c in stripped) / len(stripped)


def is_number_like(text):
    return len(text.strip()) >= MIN_LEN and digit_ratio(text) >= MIN_DIGIT_RATIO


def overlap(a_min, a_max, b_min, b_max):
    return max(0, min(a_max, b_max) - max(a_min, b_min))


def find_labels(index, items):
    """Return candidate labels to the left of and above item `index`."""
    x1, y1, x2, y2 = items[index]["box"]
    left, above = [], []
    for i, other in enumerate(items):
        if i == index:
            continue
        ox1, oy1, ox2, oy2 = other["box"]
        if ox2 <= x1 and overlap(y1, y2, oy1, oy2) > 0.5 * (y2 - y1):
            left.append((x1 - ox2, other))
        elif oy2 <= y1 and overlap(x1, x2, ox1, ox2) > 0.3 * (x2 - x1):
            above.append((y1 - oy2, other))
    left.sort(key=lambda p: p[0])
    above.sort(key=lambda p: p[0])
    return left, above


def detect(result, max_gap=400):
    items = [
        {"text": t, "box": b}
        for t, b in zip(result["rec_texts"], result["rec_boxes"])
    ]

    matches = []
    for i, item in enumerate(items):
        if not is_number_like(item["text"]):
            continue
        left, above = find_labels(i, items)
        for gap, label in left[:2] + above[:2]:
            if gap <= max_gap and KEYWORD in label["text"].lower():
                matches.append(
                    {"label": label["text"].strip(), "value": item["text"].strip()}
                )
                break
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", help="PaddleOCR *_res.json file")
    args = parser.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        result = json.load(f)

    matches = detect(result)
    print(json.dumps(matches, ensure_ascii=False, indent=2))
    return 0 if matches else 1


if __name__ == "__main__":
    sys.exit(main())
