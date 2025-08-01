# to run
# python prompt_engineer_experiment.py ../config.local.Llama-3.3-70B-Instruct.json example-request-3.jsonl output_new.json
#
#

from gpt_researcher import GPTResearcher
from gpt_researcher.utils import output 
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()
import logging
import sys
import json
import os
import time
import pandas as pd
import itertools

# TODO: make a permanent path for INSTRUCTION_JSON

# Step 1: Load Prompt Instruction Sets
# The instruction set are the list of prompt variants that will be used to generate the reports.
INSTRUCTION_JSON = "/exp/slineberger/instruction_sets.json"
num_to_generate = 2 # Number of prompts to generate for each request
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

# -- Step 2: Create Prompt Variants DataFrame
def build_prompt(combo):
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

def generate_prompt_df() -> pd.DataFrame:
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
        prompt_text = build_prompt(combo)
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

# Step 3: Custom Logs Handler
class CustomLogsHandler:
    """A custom Logs handler class to handle JSON data."""
    def __init__(self):
        self.logs = []  # Initialize logs to store data

    async def send_json(self, data: Dict[str, Any]) -> None:
        """Send JSON data and log it."""
        if data['type'] == 'logs':
            self.logs.append(data)  # Append data to logs

# Step 4: Extract Titles and Passages from Context
def extract_titles_passages(context, doc_dict):
    passages = []
    for item in context.replace('\n',' ').split('Source:'):
        fields = item.split('Content:')
        if len(fields) == 2:
            if 'Title:' in fields[0]:
                docid, title = fields[0].strip().split('Title:')
                title = title.strip()
                docid = docid.strip()
            else:
                docid = fields[0].strip()
                title = ""
                
            if docid in doc_dict:
                docid = doc_dict[docid]
            passage = fields[1].strip()
            passages.append({'docid':docid.strip(), 'title':title, 'passage':passage})
    return passages

# Step 5: Asynchronous Function to Get Report Phase 
async def get_report_phase1(query: str, background: str, report_type: str, config_file: str) -> str:
    custom_logs_handler = CustomLogsHandler()
    researcher = GPTResearcher(query, background, report_type, websocket=custom_logs_handler, config_path=config_file)

    context, doc_dict = await researcher.conduct_research()
    passages = extract_titles_passages(context, doc_dict)
    return researcher, custom_logs_handler.logs, passages, context, doc_dict


#Step 6: Main Execution Block

if __name__ == "__main__":

    report_type = "SCALE25_report"
    team_id = "hltcoe"
    task = "multilingual"

    config_file = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]
    if len(sys.argv) >= 5:
        run_name = sys.argv[4]
    else:
        run_name = f'{report_type}_{time.time()}'
    ddir = os.path.dirname(os.path.realpath(__file__))

    metadata = {'team_id':team_id, 'run_id':run_name, 'task':task}

# This is the second command-line argument (sys.argv[2]) passed to the script when it is executed. It specifies the path to the input JSONL file (e.g., example-request-3.jsonl).
    data = []
    with open(input_file) as F:
        for line in F:
            data.append(json.loads(line))



# Step 7: Generate Prompt Variants DataFrame
    prompt_df = generate_prompt_df()
    print(f"Total prompts generated: {len(prompt_df)}")
    # Optionally sample or select a subset for generation
    selected_prompts = prompt_df.sample(n=min(num_to_generate, len(prompt_df)), random_state=42)

    OUTPUT_FILES = {}
    OUTPUT_LOGS = {}
    for idx, row in selected_prompts.iterrows():
        OUTPUT_FILES[idx] = open(f'{row["id"]}.{output_file}','w')
        OUTPUT_LOGS[idx] = open(f'{row["id"]}.{output_file}.log','w')
        print(f'-------- {row["id"]} --------\n{row["prompt_text"]}')

    loop = asyncio.get_event_loop()
    for i, d in enumerate(data):
        print(f"\n============ request {i} request_id:{d['request_id']} ==============")
        query = d["problem_statement"]
        background = d["background"]

        researcher, logs, passages, context, doc_dict = loop.run_until_complete(get_report_phase1(query, background, report_type, config_file))

        for idx, row in selected_prompts.iterrows():
            print(f"\nGenerating report for prompt {row['id']}:")
            prompt_text = {}
            prompt_text["pre"] = f'''
Query: "{query}"
---
'''
            
            prompt_text["post"] = f'''
Using the above information, provide supporting facts for the query: "{background}"

{row["prompt_text"]}
'''

            start_time = time.time()

            report = loop.run_until_complete(researcher.write_report(custom_prompt=prompt_text))            
            elapsed_time = time.time() - start_time

            d['passages'] = passages
            d['logs'] = logs
            d['report'] = report
            d['doc_dict'] = doc_dict
            d['context'] = context
            metadata['topic_id'] = d['request_id']

            trec_report = output.get_trec_format(metadata, report, passages)
            OUTPUT_FILES[idx].write(json.dumps(trec_report)+'\n')

            d['report_length_by_char'] =  sum([len(t["text"]) for t in trec_report["responses"]])
            OUTPUT_LOGS[idx].write(json.dumps(d)+'\n')

            print(f"========== request {i} {row['id']} report finished in {elapsed_time:.0f} seconds; report length: {d['report_length_by_char']} chars =========\n")
            OUTPUT_LOGS[idx].flush()
            OUTPUT_FILES[idx].flush()
            sys.stdout.flush()

    for idx, row in selected_prompts.iterrows():
        OUTPUT_FILES[idx].close()
        OUTPUT_LOGS[idx].close()
