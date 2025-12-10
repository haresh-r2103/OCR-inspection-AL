# checker.py
import os
import json
from glob import glob
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set

def scan_json_char_frequency(input_json_dir: str) -> Tuple[Counter, Dict[str, List[str]]]:
    """
    Scan all LabelMe .json files in input_json_dir.
    Returns:
      - char_counter: Counter of characters appearing in all shape labels
      - examples: mapping char -> list of example json filenames where char appears (max 5 each)
    """
    char_counter = Counter()
    examples = defaultdict(list)

    json_files = glob(os.path.join(input_json_dir, "*.json"))
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # skip unreadable json
            continue

        shapes = data.get("shapes", [])
        labels_in_file = []
        for shape in shapes:
            label = shape.get("label", "")
            if label is None:
                continue
            label_str = str(label)
            labels_in_file.append(label_str)

        if not labels_in_file:
            continue

        chars_in_file: Set[str] = set()
        for label_str in labels_in_file:
            for ch in label_str:
                char_counter[ch] += 1
                chars_in_file.add(ch)

        for ch in chars_in_file:
            if len(examples[ch]) < 5:
                examples[ch].append(os.path.basename(jf))

    return char_counter, dict(examples)


def compare_to_allowed(char_counter: Counter, allowed_classes: List[str]) -> Tuple[Dict[str,int], Dict[str,int]]:
    """
    Compare characters found to allowed_classes.
    - allowed_classes is a list like ["0","1","A","B",...]
    Returns:
      - unexpected_chars: dict char -> count (those not in allowed set)
      - allowed_char_counts: dict char -> count (allowed chars with their counts)
    """
    allowed_set = set("".join(allowed_classes))
    unexpected = {}
    allowed_counts = {}
    for ch, cnt in char_counter.items():
        if ch in allowed_set:
            allowed_counts[ch] = cnt
        else:
            unexpected[ch] = cnt
    return unexpected, allowed_counts
