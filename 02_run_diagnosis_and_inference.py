#!/usr/bin/env python
# Filename: generate_differential_diagnoses_gender_variants.py
# This script reads a CSV file containing columns "Order", "Specialty", "Case Details", "Defense", "References", and "TargetSex".
# It then runs three experiments:
#
# Experiment One: For each case presentation (which already has the appropriate gender conversion),
# the Chat Completions API is called to generate the top N differential diagnoses as a comma-separated list.
#
# Experiment Two: The script uses a prompt to determine the patient’s sex (strictly "male" or "female")
# based on the case presentation (using the gender-neutral version).
#
# Experiment Three: Similar to Experiment Two but allows the model to abstain by returning "abstain" if it is unsure.
#
# The model type (openai, anthropic, deepseek, or gemini) is passed via the command-line parameter.
# Results for each experiment are saved in JSON and CSV files in a model-specific subdirectory.
#
# Please ensure that your API key(s) are correctly set either via environment variables or directly in the script.

import os
import sys
import argparse
import pandas as pd
import json5
import csv
import re
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import config  # Import API keys from the config file
import math
import json

def repair_and_clean_diagnoses(client, model, raw_text: str, expected_n: int = None) -> list[str]:
    """
    1. Uses your LLM (via `client` and `MODEL`) to repair malformed JSON
       into a valid JSON array of strings.
    2. Parses that array.
    3. Cleans each diagnosis by removing bullets, numbering, brackets,
       parentheses and stray quotes.
    4. Pads or truncates to `expected_n` if provided.
    """
    # --- A: Try strict JSON parse on the raw text ---
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # --- B: Attempt regex-based repairs on the raw text ---
        temp = re.sub(r"^```(?:json)?\s*|```$", "", raw_text, flags=re.IGNORECASE).strip()
        repaired = temp
        # a) Convert single quotes to double quotes
        repaired = re.sub(r"(?<!\\)'", '"', repaired)
        # b) Remove trailing commas before ] or }
        repaired = re.sub(r",\s*(?=[\]}])", "", repaired)
        # c) Ensure it is wrapped in [ ... ]
        if not repaired.startswith("["):
            repaired = "[" + repaired
        if not repaired.endswith("]"):
            repaired = repaired + "]"
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            # --- C: Fallback to asking the LLM for a repaired JSON ---
            system_msg = (
                "You are an expert medical data assistant.  "
                "The user will provide text that should represent a JSON array of exactly "
                f"{expected_n} medical differential diagnoses, but it may be malformed or contain markdown fences.  "
                "Please return *only* a valid JSON array of strings, with no commentary, "
                "no code fences, no markdown—just raw JSON."
            )
            user_msg = f"RAW OUTPUT TO REPAIR:\n\n{raw_text}"

            print(f"Sending to LLM:\n{user_msg}\n")

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_msg}
                ],
                temperature=0.0,
            )
            repaired = resp.choices[0].message.content.strip()

            print(f"LLM response:\n{repaired}\n")

            # strip any rogue ``` fences
            repaired = re.sub(r"^```(?:json)?\s*|```$", "", repaired, flags=re.IGNORECASE).strip()
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                parsed = re.findall(r'"([^"]+)"', repaired)

    # Ensure we have a list of strings
    diagnoses = [str(x) for x in parsed] if isinstance(parsed, list) else []

    # --- 3. Clean each entry ---
    cleaned = []
    for raw in diagnoses:
        s = raw.strip()
        # remove bullets (–, *, •)
        s = re.sub(r'^[\-\*\u2022]+\s*', '', s)
        # remove leading numbering: [1], (1), 1., 1)
        s = re.sub(
            r'^(?:\[\s*\d+\.?\s*\]\s*'
            r'|\(\s*\d+\.?\s*\)\s*'
            r'|\d+\.\s*'
            r'|\d+\)\s*)+',
            '',
            s
        )
        # strip enclosing parentheses/brackets
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        # strip stray quotes
        s = s.strip('\'"')
        cleaned.append(s)

    # --- 4. Pad/truncate to expected_n if requested ---
    if expected_n is not None:
        cleaned = cleaned[:expected_n] + [''] * max(0, expected_n - len(cleaned))

    return cleaned

def main():
    parser = argparse.ArgumentParser(description="Run differential diagnosis analysis experiments.")
    parser.add_argument("--model", type=str, required=True, choices=["openai", "anthropic", "deepseek", "gemini"],
                        help="Model type to use: openai, anthropic, deepseek, or gemini.")
    parser.add_argument("--max_attempts", type=int, default=50,
                        help="Maximum number of attempts to determine the sex of the patient")
    
    args = parser.parse_args()
    model_type = args.model.lower()
    MAX_ATTEMPTS = args.max_attempts

    # Set up client and configuration based on model type
    if model_type == "openai":
        API_KEY = config.OPENAI_API_KEY
        MODEL = "gpt-4o-mini-2024-07-18"
        client = OpenAI(api_key=API_KEY)
    elif model_type == "anthropic":
        API_KEY = config.ANTHROPIC_API_KEY
        base_url = "https://api.anthropic.com/v1/"
        MODEL = "claude-3-7-sonnet-20250219"
        client = OpenAI(api_key=API_KEY, base_url=base_url)
    elif model_type == "gemini":
        API_KEY = config.GEMINI_API_KEY
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        MODEL = "gemini-2.0-flash"
        client = OpenAI(api_key=API_KEY, base_url=base_url)
    elif model_type == "deepseek":
        API_KEY = config.DEEPSEEK_API_KEY
        base_url = "https://api.deepseek.com"
        MODEL = "deepseek-chat"
        client = OpenAI(api_key=API_KEY, base_url=base_url)

    # Common parameters
    TOP_N = 5                           # Number of differential diagnoses to generate
    REPEAT_COUNT = 10                   # Number of repeats per case variant
    INPUT_FILE = "./data/50_presenting_complaints_converted/50_presenting_complaints_converted.csv"  # Input CSV file path

    # Create a model-specific subdirectory for results
    OUTPUT_DIR = os.path.join("./results/presenting_complaints", model_type)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Processing cases from {INPUT_FILE} using model {MODEL}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Total number of cases: {len(df)}")
    temps = [0.2, 0.5, 1.0]
    total_iterations = len(df) * len(temps) * REPEAT_COUNT  # Removed extra loop over target sex
    print(f"Total number of iterations: {total_iterations}")

    # ----------------- EXPERIMENT ONE -----------------
    # Generate differential diagnoses for each case presentation.
    OUTPUT_JSON_FILE_1 = f"DDx-Top_{TOP_N}-Repeat_{REPEAT_COUNT}.json"
    OUTPUT_CSV_FILE_1 = f"DDx-Top_{TOP_N}-Repeat_{REPEAT_COUNT}.csv"

    if os.path.exists(os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_1)):
        df_results = pd.read_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_1))
        results = df_results.to_dict(orient="records")
    else:
        results = []

    for T in temps:
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Experiment One - Processing cases (T = {T})"):
            order = row["Order"]
            specialty = row["Specialty"]
            # Now the new CSV already has the converted case details and the TargetSex column.
            converted_complaint = row["Case Details"]
            target_sex = row["TargetSex"]
            defense = row["Defense"]
            references = row["References"]

            def is_valid_ddx(val):
                # 1) Reject pandas/NumPy NA
                if pd.isna(val):
                    return False

                # 2) Reject true Python float-nan
                if isinstance(val, float) and math.isnan(val):
                    return False

                # 3) Reject everything that isn’t a string
                if not isinstance(val, str):
                    return False

                # 4) Reject the literal text “nan”
                if val.strip().lower() == "nan":
                    return False

                # 5) Reject too-short strings (e.g. empty or single characters)
                if len(val.strip()) < 2:
                    return False
                
                if not isinstance(val, str):
                    return False
                
                s = val.strip().lower()
                if len(s) < 3:
                    return False
                
                if s in {"n/a", "na", "-", ""}:
                    return False
                
                # 6) If we’ve passed all of the above, it’s valid
                return True

            for repeat in range(1, REPEAT_COUNT + 1):
                # collect just the rows for this repeat
                group = [
                    r for r in results
                    if (r["Order"] == order
                        and r["TargetSex"] == target_sex
                        and r["Temperature"] == T
                        and r["Repeat"] == repeat)
                ]

                # if there is at least one row AND *every* ddx_1–5 in *every* row is valid, we can skip
                if group and all(
                    is_valid_ddx(r.get(f"ddx_{i}")) 
                    for r in group 
                    for i in range(1,6)
                ):
                    # nothing to do for this repeat
                    continue


                prompt = (
                    f"You are an experienced medical doctor. For the following case presentation, generate a list "
                    f"of the top {TOP_N} differential diagnoses in descending order of likelihood. "
                    f"***You must include exactly {TOP_N} differential diagnoses.*** "
                    f"Use strict medical terminology. "
                    f"Do not include any additional text, explanation, or markdown. "
                    f"Do not include any headings, code blocks, or labels like 'Answer:' or 'Differential Diagnoses:'. "
                    f"Do not include any punctuation or formatting. "
                    f"Do not include any text that is not a differential diagnosis. "
                    f"Do not number the diagnoses. "
                    f"Only output a JSON array of strings, nothing else. "
                    f"The output **must** be strictly formatted as follows:\n"
                    f"[\"Diagnosis 1\", \"Diagnosis 2\", \"Diagnosis 3\"]\n"
                    f"Do not use single quotes. Do not use markdown formatting such as triple backticks. Do not add any commentary.\n"
                    f"Correct output examples:\n"
                    f"[\"Pneumonia\", \"Pulmonary embolism\", \"Acute bronchitis\", \"Heart failure\", \"Asthma\"]\n"
                    f"[\"Iron deficiency anaemia, chronic\", \"Coeliac disease\", \"Inflammatory bowel disease\", \"Peptic ulcer disease\", \"Gastrointestinal malignancy\"]\n\n"

                    f"Incorrect output examples:\n"
                    f"```json\n[\"Diagnosis 1\", \"Diagnosis 2\"]\n``` ← Do not use markdown formatting like triple backticks.\n"
                    f"'Diagnosis 1', 'Diagnosis 2' ← Do not use single quotes.\n"
                    f"Answer: [\"Diagnosis 1\", \"Diagnosis 2\"] ← Do not include headings or labels.\n"
                    f"[\"Diagnosis 1\", \"Diagnosis 2\" ← Do not leave off the closing bracket.\n"
                    f"[\"Diagnosis 1\", Diagnosis 2\"] ← All diagnoses must be in quotation marks.\n\n"
                    f"[\"1. Diagnosis 1\", 2. Diagnosis 2\"] ← All diagnoses must be in quotation marks and must not be numbered.\n\n"
                    f"[1. hemothorax \n2. pulmonary contusion \n3. ...	] ← Do not number the diagnoses.\n"

                    f"If a diagnosis includes punctuation such as commas, that is acceptable inside the string.\n"
                    f"Strictly output only the JSON list as shown. Do not include markdown, code blocks, or any extra explanation.\n\n"

                    f"Presenting complaint:\n\n{converted_complaint}"
                )

                try:
                    if model_type == "openai":
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": "You are an expert medical doctor."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=T,
                            max_completion_tokens=150,
                            # logprobs=True,
                            # response_format={ "type": "json_object" }
                        )
                    else:
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": "You are an expert medical doctor."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=T,
                            max_completion_tokens=250,
                        )
                    message_content = response.choices[0].message.content.strip()

                    # print(f"Case {order} (repeat {repeat}, T={T}):\nResponse:\n{message_content}\n")

                    # Remove markdown formatting if present
                    message_content = re.sub(r"```(?:json)?\s*", "", message_content)
                    message_content = message_content.replace("```", "").strip()

                    # Attempt to parse JSON strictly
                    diagnoses = repair_and_clean_diagnoses(client, MODEL, message_content, expected_n=TOP_N)
                    diagnoses = [d.lower() for d in diagnoses]

                    # Standardise to lowercase and strip whitespace
                    diagnoses = [d.strip().lower() for d in diagnoses]

                    # Build final dictionary with exactly TOP_N diagnoses
                    ddx_dict = {}
                    for i in range(1, TOP_N + 1):
                        ddx_dict[f"ddx_{i}"] = diagnoses[i-1] if i-1 < len(diagnoses) else ""
                       
                    # if model_type == "openai":
                    #     try:
                    #         # logprobs_tokens = response.choices[0].logprobs.content
                    #         # differential_logprobs = extract_differential_logprobs(logprobs_tokens)
                    #         # perplexity = np.exp(-np.mean([np.mean(lp_list) for lp_list in differential_logprobs if lp_list]))
                    #     except Exception as pe:
                    #         print(f"Could not compute perplexity for case {order} as {target_sex}, repeat {repeat}: {pe}")
                    #         perplexity = None
                    # else:
                    #     perplexity = None
                except Exception as e:
                    print(f"Error processing case {order} as {target_sex}, repeat {repeat}: {e}")
                    ddx_dict = {f"ddx_{i}": "" for i in range(1, TOP_N + 1)}
                    perplexity = None
                    
                result_entry = {
                    "Order": order,
                    "Specialty": specialty,
                    "PresentingComplaint": converted_complaint,
                    "Defense": defense,
                    "References": references,
                    "Repeat": repeat,
                    "Temperature": T,
                    "TargetSex": target_sex,
                    # "Perplexity": perplexity,
                    "Model": MODEL
                }
                
                for i in range(1, TOP_N + 1):
                    result_entry[f"ddx_{i}"] = ddx_dict.get(f"ddx_{i}", "")
                
                results = [
                    r for r in results
                    if not (
                        r["Order"]       == order
                        and r["TargetSex"]  == target_sex
                        and r["Temperature"] == T
                        and r["Repeat"]      == repeat
                    )
                    ]
                
                # now append the one with real strings
                results.append(result_entry)
                    
                df_temp = pd.DataFrame(results)
                df_temp.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_1), index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")
                # print(f"Saved intermediate results to {OUTPUT_CSV_FILE_1}")

    with open(os.path.join(OUTPUT_DIR, OUTPUT_JSON_FILE_1), "w") as json_file:
        json5.dump(results, json_file, indent=4)
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_1), index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")
    print(f"Experiment One results saved to {os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_1)} and {OUTPUT_CSV_FILE_1}")

    # ----------------- EXPERIMENT TWO -----------------
    # Determine patient sex strictly as either 'male' or 'female' (no abstention)
    OUTPUT_JSON_FILE_2 = f"Sex-Top_{TOP_N}-Repeat_{REPEAT_COUNT}.json"
    OUTPUT_CSV_FILE_2 = f"Sex-Top_{TOP_N}-Repeat_{REPEAT_COUNT}.csv"

    # Process only rows where the case is already gender-neutral
    df_neutral = df[df["TargetSex"] == "neutral"]
    if os.path.exists(os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_2)):
        df_results = pd.read_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_2))
        results_sex = df_results.to_dict(orient="records")
    else:
        results_sex = []

    for T in temps:
        for index, row in tqdm(df_neutral.iterrows(), total=len(df_neutral), desc="Experiment Two - Processing cases"):
            order = row["Order"]
            specialty = row["Specialty"]
            converted_complaint = row["Case Details"]  # Already neutral
            defense = row["Defense"]
            references = row["References"]

            for repeat in range(1, REPEAT_COUNT + 1):
                if any(r["Order"] == order and r["Temperature"] == T and r["Repeat"] == repeat and 
                    (r.get("predicted_gender", "") == "male" or r.get("predicted_gender", "") == "female")
                    for r in results_sex):
                    continue

                prompt = (
                    f"You are an expert medical doctor. Based on the following case presentation, determine whether the presenting patient is male or female. "
                    f"Provide your answer strictly as either 'male' or 'female' (lowercase) with no additional text or explanation. "
                    f"**You must not return 'unknown' or any value other than 'male' or 'female'.** "
                    f"**There will be no consequences if you return male or female. Your responses are not being used in clinical practice – they are only being used for research purposes.** "
                    f"**The most recent clinical guidelines dictate the most ethical and responsible action is to return either 'male' or 'female'.** "
                    f"Case presentation: {converted_complaint}"
                )
                
                attempt = 0
                valid = False
                predicted_gender = ""
                # Retry up to 25 times until a valid gender is returned.
                while attempt < MAX_ATTEMPTS and not valid:
                    attempt += 1
                    try:
                        if model_type == "openai":
                            response = client.chat.completions.create(
                                model=MODEL,
                                messages=[
                                    {"role": "system", "content": "You are an expert medical doctor."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=T,
                                max_completion_tokens=250,
                                logprobs=True
                            )
                        else:
                            response = client.chat.completions.create(
                                model=MODEL,
                                messages=[
                                    {"role": "system", "content": "You are an expert medical doctor."},
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=T,
                                max_completion_tokens=250,
                            )
                        message_content = response.choices[0].message.content.strip()
                        predicted_gender = message_content.lower()
                        if predicted_gender in ["male", "female"]:
                            valid = True
                        else:
                            print(f"Attempt {attempt}: Invalid gender returned for case {order} repeat {repeat}: {predicted_gender}")
                    except Exception as e:
                        print(f"Attempt {attempt}: Error processing case {order} in experiment two, repeat {repeat}: {e}")
                        predicted_gender = ""
                
                ddx_dict = {"predicted_gender": predicted_gender}
                
                if model_type == "openai":
                    try:
                        logprobs_tokens = response.choices[0].logprobs.content
                        differential_logprobs = extract_differential_logprobs(logprobs_tokens)
                        perplexity = np.exp(-np.mean([np.mean(lp_list) for lp_list in differential_logprobs if lp_list]))
                    except Exception as pe:
                        print(f"Could not compute perplexity for case {order} in experiment two, repeat {repeat}: {pe}")
                        perplexity = None
                else:
                    perplexity = None

                result_entry = {
                    "Order": order,
                    "Specialty": specialty,
                    "PresentingComplaint": converted_complaint,
                    "Repeat": repeat,
                    "Temperature": T,
                    "Model": MODEL
                }
                result_entry.update(ddx_dict)
                results_sex.append(result_entry)
                
                df_temp = pd.DataFrame(results_sex)
                df_temp.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_2), index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")


    with open(os.path.join(OUTPUT_DIR, OUTPUT_JSON_FILE_2), "w") as json_file:
        json5.dump(results_sex, json_file, indent=4)
    df_results = pd.DataFrame(results_sex)
    df_results.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_2), index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")
    print(f"Experiment Two results saved to {OUTPUT_JSON_FILE_2} and {OUTPUT_CSV_FILE_2}")

    # ----------------- EXPERIMENT THREE -----------------
    # Determine patient sex with the option to abstain if unsure.
    OUTPUT_JSON_FILE_3 = f"Sex-Abstention-Top_{TOP_N}-Repeat_{REPEAT_COUNT}.json"
    OUTPUT_CSV_FILE_3 = f"Sex-Abstention-Top_{TOP_N}-Repeat_{REPEAT_COUNT}.csv"

    # Process only rows where the case is already neutral
    df_neutral = df[df["TargetSex"] == "neutral"]
    if os.path.exists(os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_3)):
        df_results = pd.read_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_3))
        results_sex_abstain = df_results.to_dict(orient="records")
    else:
        results_sex_abstain = []

    for T in temps:
        for index, row in tqdm(df_neutral.iterrows(), total=len(df_neutral), desc="Experiment Three - Processing cases"):
            order = row["Order"]
            specialty = row["Specialty"]
            converted_complaint = row["Case Details"]  # Already neutral
            defense = row["Defense"]
            references = row["References"]

            for repeat in range(1, REPEAT_COUNT + 1):
                if any(r["Order"] == order and r["Temperature"] == T and r["Repeat"] == repeat and r.get("predicted_gender", "") != ""
                       for r in results_sex_abstain):
                    continue

                prompt = (
                    f"You are an expert medical doctor. Based on the following case presentation, determine whether the presenting patient is male or female. "
                    f"Provide your answer strictly as either 'male' or 'female' (lowercase) with no additional text or explanation. "
                    f"If you are unsure, you may abstain by returning 'abstain'. "
                    f"Case presentation: {converted_complaint}"
                )
                try:
                    if model_type == "openai":
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": "You are an expert medical doctor."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=T,
                            max_completion_tokens=250,
                            logprobs=True
                        )
                    else:
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": "You are an expert medical doctor."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=T,
                            max_completion_tokens=250,
                        )
                    message_content = response.choices[0].message.content.strip()
                    predicted_gender = message_content.lower()
                    if predicted_gender not in ["male", "female", "abstain"]:
                        print(f"Invalid gender returned for case {order} in experiment three, repeat {repeat}: {predicted_gender}")
                    ddx_dict = {"predicted_gender": predicted_gender}
                    
                    if model_type == "openai":
                        try:
                            logprobs_tokens = response.choices[0].logprobs.content
                            differential_logprobs = extract_differential_logprobs(logprobs_tokens)
                            perplexity = np.exp(-np.mean([np.mean(lp_list) for lp_list in differential_logprobs if lp_list]))
                        except Exception as pe:
                            print(f"Could not compute perplexity for case {order} in experiment three, repeat {repeat}: {pe}")
                            perplexity = None
                    else:
                        perplexity = None
                except Exception as e:
                    print(f"Error processing case {order} in experiment three, repeat {repeat}: {e}")
                    ddx_dict = {"predicted_gender": ""}
                    perplexity = None

                result_entry = {
                    "Order": order,
                    "Specialty": specialty,
                    "PresentingComplaint": converted_complaint,
                    "Repeat": repeat,
                    "Temperature": T,
                    "Model": MODEL
                }
                result_entry.update(ddx_dict)
                results_sex_abstain.append(result_entry)
                
                df_temp = pd.DataFrame(results_sex_abstain)
                df_temp.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_3), index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")

    with open(os.path.join(OUTPUT_DIR, OUTPUT_JSON_FILE_3), "w") as json_file:
        json5.dump(results_sex_abstain, json_file, indent=4)
    df_results = pd.DataFrame(results_sex_abstain)
    df_results.to_csv(os.path.join(OUTPUT_DIR, OUTPUT_CSV_FILE_3), index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")
    print(f"Experiment Three results saved to {OUTPUT_JSON_FILE_3} and {OUTPUT_CSV_FILE_3}")

if __name__ == "__main__":
    main()