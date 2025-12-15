#!/usr/bin/env python3
"""
generate_synonym_dictionary.py

This script examines all differential diagnoses generated for each presenting complaint
across all temperatures and all models (OpenAI, Anthropic, DeepSeek, and Gemini).
Then, for each presenting complaint, it prompts the OpenAI API to produce
a strict, medically accurate synonym dictionary.

- The synonyms must mean the EXACT same thing medically.
- If there is ambiguity, or if two terms are not precisely the same condition,
  they should NOT be unified.

Output:
- For each presenting complaint (or "Order"), the script saves a JSON file named:
  synonym_dict_order_<ORDER_ID>.json
  into a specified directory (by default, ./analysis/synonym_dictionaries).
"""

import os
import json
import re
import pandas as pd
from pathlib import Path

# Import your existing config for the OpenAI API key
import config
from openai import OpenAI

########################################################################
#                           CONFIGURATION                              #
########################################################################

# Four subdirectories containing the main CSV results
MODEL_DIRS = ["openai", "anthropic", "deepseek", "gemini"]

# Name of the CSV file in each model directory
CSV_FILENAME = "DDx-Top_5-Repeat_10.csv"

# Where to find the results subdirectories
RESULTS_BASE_DIR = Path("./results/presenting_complaints")

# Where to store the final synonym dictionary JSON files
OUTPUT_SYNONYM_DIR = Path("./analysis/synonym_dictionaries")
OUTPUT_SYNONYM_DIR.mkdir(parents=True, exist_ok=True)

# Which OpenAI model to use for generating the dictionary
OPENAI_MODEL_NAME = "gpt-4o-mini-2024-07-18"

########################################################################
#                  SETUP THE OPENAI CLIENT (EXAMPLE)                   #
########################################################################

# If you have a custom domain or base_url, set it here if needed
# For example: base_url = "https://api.openai.com/v1/"
base_url = None  

# Instantiate the OpenAI client
openai_api_key = config.OPENAI_API_KEY
client = OpenAI(api_key=openai_api_key, base_url=base_url)

########################################################################
#                         HELPER FUNCTIONS                             #
########################################################################

def collect_all_ddx():
    """
    Reads each model directory, loads the CSV file, and concatenates them
    into a single DataFrame. Returns this combined DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all rows from all four LLM CSV files,
        with columns including Order, PresentingComplaint, ddx_1..ddx_5, etc.
    """
    all_dfs = []
    for mdir in MODEL_DIRS:
        csv_path = RESULTS_BASE_DIR / mdir / CSV_FILENAME
        if csv_path.exists():
            df_temp = pd.read_csv(csv_path)
            # Optional: store which model's CSV this row came from
            if "SourceModel" not in df_temp.columns:
                df_temp["SourceModel"] = mdir
            all_dfs.append(df_temp)
        else:
            print(f"Warning: CSV file not found for model directory {mdir}: {csv_path}")

    if len(all_dfs) == 0:
        raise FileNotFoundError("No CSV files were found in the specified directories.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


#!/usr/bin/env python3
"""
generate_synonym_dictionary.py

Updated version that avoids truncation / partial JSON errors by:
1. Chunking large diagnosis lists into smaller sets (e.g., 100 diagnoses per chunk).
2. Prompting GPT-4 for each chunk separately.
3. Merging the chunk-level synonym dictionaries into one final dictionary.

Author: Your Name
Date: YYYY-MM-DD
"""

import os
import json
import pandas as pd
from pathlib import Path

import config  # Load your OPENAI_API_KEY from here
from openai import OpenAI

########################################################################
#                           CONFIGURATION                              #
########################################################################

# Four subdirectories containing the main CSV results
MODEL_DIRS = ["openai", "anthropic", "deepseek", "gemini"]

# Name of the CSV file in each model directory
CSV_FILENAME = "DDx-Top_5-Repeat_10.csv"

# Where to find the results subdirectories
RESULTS_BASE_DIR = Path("./results/presenting_complaints")

# Where to store the final synonym dictionary JSON files
OUTPUT_SYNONYM_DIR = Path("./analysis/synonym_dictionaries")
OUTPUT_SYNONYM_DIR.mkdir(parents=True, exist_ok=True)

# Which OpenAI model to use for generating the dictionary
OPENAI_MODEL_NAME = "gpt-4o-mini-2024-07-18"

########################################################################
#                  SETUP THE OPENAI CLIENT (EXAMPLE)                   #
########################################################################

openai_api_key = config.OPENAI_API_KEY
client = OpenAI(api_key=openai_api_key)

########################################################################
#                         HELPER FUNCTIONS                             #
########################################################################

def collect_all_ddx() -> pd.DataFrame:
    """
    Reads each model directory, loads the CSV file, and concatenates them
    into a single DataFrame. Returns this combined DataFrame.
    """
    all_dfs = []
    for mdir in MODEL_DIRS:
        csv_path = RESULTS_BASE_DIR / mdir / CSV_FILENAME
        if csv_path.exists():
            df_temp = pd.read_csv(csv_path)
            if "SourceModel" not in df_temp.columns:
                df_temp["SourceModel"] = mdir
            all_dfs.append(df_temp)
        else:
            print(f"Warning: CSV file not found for {mdir}: {csv_path}")

    if len(all_dfs) == 0:
        raise FileNotFoundError("No CSV files were found in the specified directories.")

    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df


def chunk_list(lst, chunk_size=100):
    """
    Yield successive chunk_size-sized chunks from lst.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def prompt_for_synonyms(diagnoses_chunk):
    """
    Sends one chunk of diagnoses to GPT-4 and returns a dictionary
    mapping each diagnosis to its canonical name.

    We keep the prompt short, strict, and include examples to ensure valid JSON.
    """
    system_message = (
        "You are an expert medical doctor. You are extremely precise about medical terminology."
        " Never return code blocks, markdown fences, or extra commentary."
    )

    # Provide a short example so the model sees exactly what we want
    # We also strongly request no extraneous text.
    user_prompt = f"""
You are given a list of medical diagnoses that all relate to the same presenting complaint.
Some might be synonyms for the exact same condition.

1. Only unify terms if they are 100% medically identical.
2. If they differ, do not unify.
3. Output ONLY valid JSON. No code fences, no additional commentary.
4. For any two (or more) synonyms, pick the single best canonical name (the most standard term).
5. If a term is already the canonical name, map it to itself.

Example format (no markdown, only JSON):
{{
  "unstable angina pectoris": "Unstable Angina",
  "u.a. pectoris": "Unstable Angina",
  "unstable angina": "Unstable Angina"
}}

Now process the following list of diagnoses. Return strict JSON only:
{diagnoses_chunk}
"""

    # Call the model
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt.strip()},
            ],
            temperature=0.0,  # ensure a more deterministic response
            max_completion_tokens=2000  # Increase if needed
        )
    except Exception as e:
        print(f"OpenAI API call failed for chunk: {e}")
        return {}

    content = response.choices[0].message.content.strip()

    # Attempt to parse as JSON
    try:
        chunk_dict = json.loads(content)
        return chunk_dict
    except json.JSONDecodeError:
        print("Could not decode JSON for chunk, returning empty dictionary.")
        # Debugging: you could print(content) or log partial content
        return {}


def merge_dictionaries(master_dict, new_dict):
    """
    Merges 'new_dict' entries into 'master_dict'.
    If new_dict has a key that doesn't exist in master_dict, add it.
    If it does exist, override or keep existing? We'll keep the new mapping
    if we want the most recent approach. Or just keep the old one.

    For simplicity, we always use new_dict's value for any key collisions.
    """
    for k, v in new_dict.items():
        master_dict[k] = v
    return master_dict


def create_synonym_dictionary_for_complaint(order_id, presenting_complaint, diagnoses, chunk_size=100):
    """
    Chunk-based approach to unify synonyms for a single presenting complaint.
    1. Split the diagnoses into chunks of 'chunk_size'.
    2. For each chunk, prompt GPT-4 for a synonyms dictionary.
    3. Merge those chunk dictionaries into one 'master_dict'.
    4. Return the resulting dictionary.
    """
    unique_diagnoses = list(set(diagnoses))
    unique_diagnoses.sort()
    master_dict = {}

    for chunk in chunk_list(unique_diagnoses, chunk_size=chunk_size):
        chunk_dict = prompt_for_synonyms(chunk)
        master_dict = merge_dictionaries(master_dict, chunk_dict)

    return master_dict

########################################################################
#                         MAIN SCRIPT LOGIC                            #
########################################################################

if __name__ == "__main__":
    # 1. Collect all ddx data from the four LLM CSVs
    df_all = collect_all_ddx()

    # 2. Group by "Order" so each complaint is processed individually.
    grouped = df_all.groupby("Order", dropna=False)

    for order_id, group_df in grouped:
        # File name for synonyms
        output_file = OUTPUT_SYNONYM_DIR / f"synonym_dict_order_{order_id}.json"

        # If the file already exists, skip
        if output_file.exists():
            print(f"Synonym file already exists for order={order_id}; skipping.")
            continue

        presenting_complaint = str(group_df["PresentingComplaint"].iloc[0])
        ddx_cols = [c for c in group_df.columns if c.startswith("ddx_")]
        all_diags = []
        for col in ddx_cols:
            col_vals = group_df[col].dropna().tolist()
            col_vals = [str(v).strip() for v in col_vals if str(v).strip()]
            all_diags.extend(col_vals)

        if not all_diags:
            print(f"No diagnoses found for order={order_id}. Skipping.")
            continue

        print(f"\nGenerating synonym dictionary for Order={order_id} ({len(all_diags):,} total mentions).")

        # 3. Create the chunk-based synonym dictionary
        synonyms_dict = create_synonym_dictionary_for_complaint(
            order_id=order_id,
            presenting_complaint=presenting_complaint,
            diagnoses=all_diags,
            chunk_size=100  # adjust chunk size if needed
        )

        # 4. Save to JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(synonyms_dict, f, indent=2, ensure_ascii=False)

        print(f"Saved synonyms for order={order_id} to {output_file}")

    print("\nAll presenting complaints processed. Synonym dictionaries are in:", OUTPUT_SYNONYM_DIR)




########################################################################
#                         MAIN SCRIPT LOGIC                            #
########################################################################

if __name__ == "__main__":
    # 1. Collect all ddx data from the four LLM CSVs
    df_all = collect_all_ddx()

    # 2. Group by "Order" (or by "PresentingComplaint") so that each unique
    #    presenting complaint is handled individually.
    #    Often "Order" is a unique identifier, but if you prefer grouping
    #    purely by the text "PresentingComplaint," adjust accordingly.
    grouped = df_all.groupby("Order", dropna=False)

    # 3. For each group, gather all ddx_ columns. Then call the OpenAI approach
    #    to unify synonyms. Finally, save the resulting dictionary.
    for order_id, group_df in grouped:
        # We'll grab the first row's PresentingComplaint for reference
        # (assuming all rows in the group have the same complaint text).
        presenting_complaint = group_df["PresentingComplaint"].iloc[0]

        # If you want to be safe about special characters in file names,
        # you can sanitize the complaint text or just rely on order_id for uniqueness.
        # We'll produce: synonym_dict_order_<order_id>.json
        output_file = OUTPUT_SYNONYM_DIR / f"synonym_dict_order_{order_id}.json"
        if output_file.exists():
            print(f"Synonym file already exists for order={order_id}; skipping.")
            continue

        # Gather all unique diagnoses
        ddx_cols = [c for c in group_df.columns if c.startswith("ddx_")]
        all_diags = []
        for col in ddx_cols:
            # Extend with all non-null diagnoses from this column
            col_vals = group_df[col].dropna().tolist()
            # Convert them to strings (some might already be strings)
            col_vals = [str(v).strip() for v in col_vals if str(v).strip()]
            all_diags.extend(col_vals)

        if not all_diags:
            print(f"No diagnoses found for order={order_id}. Skipping.")
            continue

        print(f"\nGenerating synonym dictionary for Order={order_id} ({len(all_diags):,} total mentions).")

        # 4. Create the synonym dictionary with the OpenAI prompt
        synonyms_dict = create_synonym_dictionary_for_complaint(
            order_id=order_id,
            presenting_complaint=presenting_complaint,
            diagnoses=all_diags
        )

        # 5. Save the resulting synonyms dictionary to a JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(synonyms_dict, f, indent=2, ensure_ascii=False)

        print(f"Saved synonyms for order={order_id} to {output_file}")
    
    print("\nAll presenting complaints processed. Synonym dictionaries are in:", OUTPUT_SYNONYM_DIR)
