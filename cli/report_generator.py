import sys
import asyncio
import time
from datetime import datetime
import json
import re

from gpt_researcher import GPTResearcher
from gpt_researcher.utils.llm import generic_prompt_call

class CustomLogsHandler:
    """
    A custom Logs handler class to handle JSON data.
    """
    def __init__(self):
        self.logs = []
    async def send_json(self, data):
        if data.get('type') == 'logs':
            self.logs.append(data)

async def get_report(query, background, report_type, config_file):
    semaphore = asyncio.Semaphore(1)
    custom_logs_handler = CustomLogsHandler()
    researcher = GPTResearcher(query, background, report_type, websocket=custom_logs_handler, config_path=config_file)

    async with semaphore:
        context, doc_dict = await researcher.conduct_research()
        passages = extract_titles_passages(context, doc_dict)
        raw_report = await researcher.write_report()

    # Construct the report in the desired format
    report = {
        "metadata": {
            "team_id": "hltcoe",
            "run_id": "example_run_id",
            "topic_id": query
        },
        "responses": [
            {
                "text": passage["passage"],
                "citations": {passage["docid"]: 1.0}
            }
            for passage in passages
        ],
        "references": list(set(passage["docid"] for passage in passages))
    }

    return report, custom_logs_handler.logs, passages, context, doc_dict

async def write_reports(output_file, data, config_file, report_type):
    with open(output_file + '.log', 'w') as LOG, open(output_file, 'w') as OUT:
        for i, d in enumerate(data):
            print(f"\n============ request {i} request_id:{d['request_id']} ==============")
            query = d["problem_statement"]
            background = d["background"]
            start_time = time.time()

            report, logs, passages, context, doc_dict = await get_report(query, background, report_type, config_file)

            elapsed_time = time.time() - start_time
            print(f"========== request {i} finished in {elapsed_time:.0f} seconds =========\n")

            d['passages'] = passages
            d['logs'] = logs
            d['report'] = report

            LOG.write(json.dumps(d) + '\n')
            OUT.write(json.dumps(report, indent=2) + '\n')
            LOG.flush()
            OUT.flush()
            sys.stdout.flush()

def extract_titles_passages(context, doc_dict):
    """
    Placeholder for passage extraction logic.
    """
    # You will need to implement the actual logic as in your original script.
    # For now, return an empty list.
    return []