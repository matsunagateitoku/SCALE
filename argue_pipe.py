"""
Integrated Workflow for Research, Prompt Generation, Report Creation, and Argue-Eval Scoring

This script combines:
- Research context loading
- Prompt variant dataframe creation
- Report generation from prompts
- Automatic evaluation with Argue-Eval (judge/evaluate_report)

Requirements:
- gpt_researcher and argue_eval installed and importable
- instruction_sets.json, config JSON, and research context available
- External dependencies: pandas, asyncio, transformers, deprecated

Usage:
- Update all paths as needed (config, instruction_sets, context, doc_dict, etc).
- This script is meant to be run as a .py file or in a notebook cell.
"""

import os
import json
import itertools
import asyncio
import pandas as pd
from datetime import date
from typing import Dict, Any, List, Tuple

# -- Step 1: Import GPTResearcher and Argue-Eval
from gpt_researcher import GPTResearcher
from gpt_researcher.utils.llm import generic_prompt_call
from argue_eval.judge import evaluate_report, ModelProvider

# -- Step 2: Load Config and Research Context
CONFIG_PATH = "./config.dawn.Llama-3.3-70B-Instruct.json"
INSTRUCTION_JSON = "instruction_sets.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
print("Loaded Configuration:", config)

# Example research context (replace with actual content)
context = '...'  # Place your context string here
doc_dict = {...} # Place your doc_dict here

# -- Step 3: Load Prompt Instruction Sets
with open(INSTRUCTION_JSON, "r") as f:
    instructions = json.load(f)

total_words_options = instructions["total_words_options"]
focus_instructions = instructions["focus_instructions"]
structure_instructions = instructions["structure_instructions"]
fact_instructions = instructions["fact_instructions"]
length_instructions = instructions["length_instructions"]
depth_instructions = instructions["depth_instructions"]
opinion_instructions = instructions["opinion_instructions"]
bias_instructions = instructions["bias_instructions"]
citation_instructions = instructions["citation_instructions"]
tone_instructions = instructions["tone_instructions"]
formatting_instructions = instructions["formatting_instructions"]
extra_instructions = instructions["extra_instructions"]

# -- Step 4: Create Prompt Variants DataFrame

def build_prompt(combo, query, background):
    (
        total_words,
        focus_instruction,
        structure_instruction,
        fact_instruction,
        length_instruction,
        depth_instruction,
        opinion_instruction,
        bias_instruction,
        citation_instruction,
        tone_instruction,
        formatting_instruction,
        extra_instruction
    ) = combo

    prompt = f"""
Information: "{query}"
---

Using the above information, provide supporting facts for the query: "{background}"

Report Requirements:
- {focus_instruction}
- {structure_instruction}
- {fact_instruction}
- {length_instruction.replace('{total_words}', str(total_words))}
- {depth_instruction}
- {opinion_instruction}
- {bias_instruction}
- {citation_instruction}
- {tone_instruction}
- {formatting_instruction}
- {extra_instruction}
""".strip()
    return prompt

def generate_prompt_df(query: str, background: str) -> pd.DataFrame:
    all_combinations = list(itertools.product(
        total_words_options,
        focus_instructions,
        structure_instructions,
        fact_instructions,
        length_instructions,
        depth_instructions,
        opinion_instructions,
        bias_instructions,
        citation_instructions,
        tone_instructions,
        formatting_instructions,
        extra_instructions
    ))
    prompt_variants = []
    for i, combo in enumerate(all_combinations):
        prompt_text = build_prompt(combo, query, background)
        prompt_variants.append({
            "id": f"prompt_{i+1}",
            "total_words": combo[0],
            "focus_instruction": combo[1],
            "structure_instruction": combo[2],
            "fact_instruction": combo[3],
            "length_instruction": combo[4],
            "depth_instruction": combo[5],
            "opinion_instruction": combo[6],
            "bias_instruction": combo[7],
            "citation_instruction": combo[8],
            "tone_instruction": combo[9],
            "formatting_instruction": combo[10],
            "extra_instruction": combo[11],
            "prompt_text": prompt_text
        })
    prompt_df = pd.DataFrame(prompt_variants)
    return prompt_df

# -- Step 5: Main Research/Prompt/Report/Eval Pipeline

class CustomLogsHandler:
    """A custom Logs handler class to handle JSON data."""
    def __init__(self):
        self.logs = []
    async def send_json(self, data: Dict[str, Any]) -> None:
        if data.get('type') == 'logs':
            self.logs.append(data)

async def research_prompt_report_eval(
    query: str,
    background: str,
    report_type: str,
    num_to_generate: int = 5
) -> pd.DataFrame:
    # 1. Generate prompts DataFrame
    prompt_df = generate_prompt_df(query, background)
    print(f"Total prompts generated: {len(prompt_df)}")
    # Optionally sample or select a subset for generation
    selected_prompts = prompt_df.sample(n=min(num_to_generate, len(prompt_df)), random_state=42)
    # 2. Initialize researcher
    custom_logs_handler = CustomLogsHandler()
    researcher = GPTResearcher(query, background, report_type, websocket=custom_logs_handler, config_path=CONFIG_PATH)
    # 3. Generate reports for selected prompts
    reports = []
    for idx, row in selected_prompts.iterrows():
        print(f"\nGenerating report for prompt {row['id']}:")
        prompt_text = row["prompt_text"]
        response = await generic_prompt_call(
            agent_role_prompt=None,
            user_prompt=prompt_text,
            cfg=researcher.cfg,
            websocket=researcher.websocket,
            cost_callback=researcher.add_costs,
            step="relevance"
        )
        reports.append(response)
    selected_prompts = selected_prompts.copy()
    selected_prompts["generated_report"] = reports
    # 4. Evaluate with Argue-Eval
    # You may need to adjust the call depending on your installation/environment
    selected_prompts["argue_eval"] = [
        evaluate_report(report, ModelProvider.AUTO) for report in selected_prompts["generated_report"]
    ]
    return selected_prompts

# -- Step 6: Example Usage

if __name__ == "__main__":
    # Example research question and background
    query = "I need a report about Condoleezza Rice that focuses on her time in public service and that details specific statements or activities by her that either demonstrated her commitment to U.S. democratic values, showcased her efforts to promote democracy worldwide, or that had a significant impact on global affairs."
    background = "I am a high school student writing a report for an assignment about influential women in politics that will focus on Condoleezza Rice."
    report_type = "SCALE25_report2"

    # Run full workflow (async)
    results_df = asyncio.run(
        research_prompt_report_eval(
            query=query,
            background=background,
            report_type=report_type,
            num_to_generate=5  # Set how many prompt/report/evals to run
        )
    )
    # Save or examine the resulting DataFrame
    results_df.to_csv("reports_with_argue_eval.csv", index=False)
    print("Saved results to reports_with_argue_eval.csv")
