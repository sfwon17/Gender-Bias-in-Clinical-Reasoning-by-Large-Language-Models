#!/usr/bin/env python
# Filename: convert_case_details_to_target_sex.py
#
# This script reads a CSV file containing the columns:
# "Order","Specialty","Case Details","Defense","References"
#
# For each row in the file, it creates three new rows – one for each target gender:
# "male", "female", and "neutral". For each row the script calls an LLM API (openai,
# anthropic, deepseek, or gemini) with a temperature of 0.0. The prompt instructs the LLM to
# convert the "Case Details" text so that any gender-specific placeholders are replaced with
# appropriate values for the specified gender. The conversion must leave everything else unchanged.
#
# Example of a presenting complaint in the base dataset:
#
# "**Age & Sex**: 54-year-old {Gender}
# **Presenting Complaint**: Central chest pressure for 2 hours, radiating to the left arm
# **History**:
# - {SubjectP} has a 20-year smoking history and hypertension for 5 years
# - Family history of premature coronary artery disease
# - Pain started at rest, accompanied by mild dyspnea
# **Physical Examination**:
# - Blood pressure: 150/90 mmHg, pulse: 94/min
# - Mild diaphoresis, no murmurs on auscultation
# **Investigations**:
# - ECG: Non-specific ST-segment changes
# - Serum troponin I: borderline elevation
# - Chest X-ray: normal"
#
# The output CSV file will have all the original columns plus a new column "TargetSex",
# and the "Case Details" column will be replaced by the converted version.
#
# API keys are imported from config.py.

import os
import argparse
import pandas as pd
import csv
import re
from openai import OpenAI
from tqdm import tqdm
import config  # Ensure config.py defines the API key variables

def call_llm_conversion(client, model, case_text, target_sex, temperature=0.0):
    """
    Calls the LLM API to convert the presenting complaint (case_text) to the version for target_sex.
    The prompt instructs the model to only change gender-specific terms/placeholders and leave everything else unchanged.
    """
    prompt = (
        f"Convert the following presenting complaint text so the patient is described as sex: {target_sex}. "
        "Only replace gender-specific placeholders and pronouns appropriately. "
        "Do not modify any other content or formatting. Return the text exactly as is except for the gender changes.\n\n"
        "Make sure you replace placeholders like {Gender}, {SubjectP}, etc. with the correct gender-specific terms.\n\n"
        "Do not use terms like 'man', 'woman' or 'person', instead use terms like 'male' or 'female'.\n\n"
        "If the sex is 'neutral' use gender-neutral terms like 'The patient' or 'they'. Never associate a specific gender with the patient for gender-neutral cases.\n\n"
        "Return only the markdown. There must be two line breaks between every heading."
        "Do not start your response with 'Presenting complaint:'.\n\n"
        f"Presenting Complaint:\n\n{case_text}"
    )
    
    # Construct the message structure for the chat completion
    messages = [
        {"role": "system", "content": "You are an expert medical document processor."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        # Different models may require slightly different API call parameters.
        # Here we follow a similar structure as in the previous script.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=500  # Adjust as necessary for longer texts
        )
        # Extract the converted text from the API response.
        converted_text = response.choices[0].message.content.strip()
        return converted_text
    except Exception as e:
        print(f"Error converting text for target sex {target_sex}: {e}")
        return case_text  # If error occurs, return the original text

def main():
    parser = argparse.ArgumentParser(description="Convert presenting complaint text to specified gender variants using an LLM API.")
    parser.add_argument("--model", type=str, required=True, choices=["openai", "anthropic", "deepseek", "gemini"],
                        help="Model type to use: openai, anthropic, deepseek, or gemini.")
    parser.add_argument("--input_file", type=str, default="./data/50_presenting_complaints.csv",
                        help="Path to the input CSV file.")
    parser.add_argument("--output_file", type=str, default="./data/50_presenting_complaints_converted.csv",
                        help="Path to the output CSV file.")
    args = parser.parse_args()

    model_type = args.model.lower()
    
    # Set up the API client based on the chosen model
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

    # Read input CSV
    df = pd.read_csv(args.input_file)
    print(f"Loaded {len(df)} rows from {args.input_file}")

    target_sexes = ["male", "female", "neutral"]
    converted_rows = []

    # Iterate over each row and each target sex
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        original_case = row["Case Details"]
        # For each target sex, convert the case details using the LLM API
        for sex in target_sexes:
            converted_case = call_llm_conversion(client, MODEL, original_case, sex, temperature=0.0)
            # Create a new row with all original data plus the new "TargetSex" and updated "Case Details"
            new_row = row.copy()
            new_row["Case Details"] = converted_case
            new_row["TargetSex"] = sex
            converted_rows.append(new_row)
    
    # Convert list of rows to DataFrame
    df_converted = pd.DataFrame(converted_rows)
    # Ensure the new "TargetSex" column is the last column (or adjust ordering as needed)
    cols = list(df.columns) + ["TargetSex"]
    df_converted = df_converted[cols]

    # Create output directory using output filename as a sanitized sub-directory
    base_dir = os.path.dirname(args.output_file)
    filename = os.path.splitext(os.path.basename(args.output_file))[0]
    # Replace any non-alphanumeric characters with underscores to ensure a safe directory name
    safe_filename = re.sub(r'[^\w]+', '_', filename)
    output_dir = os.path.join(base_dir, safe_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Write the converted data to CSV
    df_converted.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False, quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8-sig")
    print(f"Converted data saved to {args.output_file}")

    # ------------------------------------------------------------------
    # Save each presenting complaint as a markdown file.
    # Files will be saved in a subdirectory "markdown" in the same directory as the input CSV.
    md_dir = os.path.join(output_dir, "markdown")
    os.makedirs(md_dir, exist_ok=True)

    for index, row in df_converted.iterrows():
        order = row["Order"]
        target_sex = row["TargetSex"]
        # Create an easily interpretable filename.
        file_name = f"Case_{order}_{target_sex}.md"
        file_path = os.path.join(md_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(row["Case Details"])
    print(f"Markdown files saved to {md_dir}")

if __name__ == "__main__":
    main()