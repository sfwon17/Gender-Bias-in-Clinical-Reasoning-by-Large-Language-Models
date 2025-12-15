#!/usr/bin/env python3
"""
collate_synonym_dictionaries.py

This script collates all order-specific synonym dictionaries created by
'generate_synonym_dictionary.py' into a single global dictionary file.

Usage:
  python collate_synonym_dictionaries.py

What it does:
1. Looks in the OUTPUT_SYNONYM_DIR (by default ./analysis/synonym_dictionaries)
   for any file named synonym_dict_order_<ORDER_ID>.json.
2. Loads each JSON file into memory.
3. Merges them into one "global" dictionary that maps every variant to a canonical name.
4. Saves that final dictionary to 'synonym_dict_all.json' in the same directory.

"""

import os
import json
from pathlib import Path

# Adjust if you changed the location in your original script
OUTPUT_SYNONYM_DIR = Path("./analysis/synonym_dictionaries")
GLOBAL_OUTPUT_FILE = OUTPUT_SYNONYM_DIR / "synonym_dict_all.json"

def merge_dictionaries(master_dict, new_dict):
    """
    Merges 'new_dict' entries into 'master_dict'.
    If new_dict has a key that doesn't exist in master_dict, add it.
    If it does exist, we override with new_dict's value for simplicity.
    """
    for k, v in new_dict.items():
        master_dict[k] = v
    return master_dict

def main():
    if not OUTPUT_SYNONYM_DIR.is_dir():
        print(f"Directory not found: {OUTPUT_SYNONYM_DIR}")
        return

    # Find all JSON files matching 'synonym_dict_order_*.json'
    order_files = list(OUTPUT_SYNONYM_DIR.glob("synonym_dict_order_*.json"))
    if not order_files:
        print(f"No per-order JSON files found in {OUTPUT_SYNONYM_DIR}.")
        return

    print(f"Found {len(order_files)} order-specific dictionary files in {OUTPUT_SYNONYM_DIR}.")

    global_synonyms_dict = {}

    # Load each file and merge into global_synonyms_dict
    for fpath in order_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                order_dict = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON from {fpath}. Skipping.")
            continue
        
        global_synonyms_dict = merge_dictionaries(global_synonyms_dict, order_dict)

    # Save the final merged dictionary
    with open(GLOBAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(global_synonyms_dict, f, indent=2, ensure_ascii=False)

    print(f"Merged dictionary saved to {GLOBAL_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
