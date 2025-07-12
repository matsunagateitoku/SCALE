import asyncio
import pandas as pd
import json
import re
from datetime import datetime

from config_loader import load_config, load_instructions
from prompt_generator import generate_prompt_df
from report_generator import CustomLogsHandler
from report_generator import extract_titles_passages
from gpt_researcher import GPTResearcher
from gpt_researcher.utils.llm import generic_prompt_call
from evaluator import run_evaluation

CONFIG_PATH = "./config.dawn.Llama-3.3-70B-Instruct.json"
INSTRUCTION_JSON = "instruction_sets.json"
NUGGETS_PATH = "/home/hltcoe/slineberger/workspace/generation/argue-eval/data/nuggets/sample_nuggets_388.json"

async def research_prompt_report_eval(
    query: str,
    background: str,
    report_type: str,
    num_to_generate: int = 5
) -> pd.DataFrame:
    instructions = load_instructions(INSTRUCTION_JSON)
    prompt_df = generate_prompt_df(query, background, instructions)
    print(f"Total prompts generated: {len(prompt_df)}")
    selected_prompts = prompt_df.sample(n=min(num_to_generate, len(prompt_df)), random_state=42)

    custom_logs_handler = CustomLogsHandler()
    researcher = GPTResearcher(query, background, report_type, websocket=custom_logs_handler, config_path=CONFIG_PATH)

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
        print(f"Raw response for prompt {row['id']}:\n{response}")

        try:
            response_dict = {
                "metadata": {
                    "team_id": "scott's prompt team",
                    "run_id": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    "topic_id": row["id"]
                },
                "responses": [],
                "references": []
            }

            citation_pattern = r"\[Source: (\d+)\]"
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response)
            for sentence in sentences:
                if sentence.strip():
                    citations = re.findall(citation_pattern, sentence)
                    response_dict["responses"].append({
                        "text": sentence.strip(),
                        "citations": {f"Source {citation}": 1.0 for citation in citations}
                    })
                    for citation in citations:
                        reference = f"Source {citation}"
                        if reference not in response_dict["references"]:
                            response_dict["references"].append(reference)

        except Exception as e:
            print(f"Error processing response for prompt {row['id']}: {e}")
            response_dict = {
                "metadata": {"team_id": "unknown", "run_id": "unknown", "topic_id": row["id"]},
                "responses": [],
                "references": []
            }

        reports.append(response_dict)

        def pretty_print_response(response_dict):
            formatted = json.dumps(response_dict, indent=2)
            if "responses" in response_dict:
                formatted = formatted.replace('}, {', '},\n{')
            return formatted

        formatted_report_with_line_breaks = pretty_print_response(response_dict)
        print(formatted_report_with_line_breaks)

    selected_prompts["generated_report"] = reports

    argue_eval_results = []
    for report in selected_prompts["generated_report"]:
        print(f"Validated report: {json.dumps(report, indent=2)}")
        result = await run_evaluation(
            report,
            NUGGETS_PATH,
            provider="hltcoe_local"
        )
        argue_eval_results.append(result)

        team_id = result["team_id"]
        topic_id = result["topic_id"]
        run_id = result["run_id"]

        for segment in result["segments"]:
            text = segment["text"]
            judgments = segment["judgments"]
            print(f"Text: {text}")
            for judgment in judgments:
                print(f"  Judgment Type: {judgment['judgment_type_id']}")
                print(f"  Response: {judgment['response']}")
                print(f"  Evaluator: {judgment['evaluator']}")

            for response in report["responses"]:
                response["scores"] = [
                    {
                        "judgment_type": "example_judgment",
                        "response": {"example_key": "example_value"},
                        "evaluator": "example_evaluator"
                    }
                ]

    selected_prompts["argue_eval"] = argue_eval_results
    return selected_prompts

def main():
    query = "I need a report about Condoleezza Rice that focuses on her time in public service and that details specific statements or activities by her that either demonstrated her commitment to U.S. democratic values, showcased her efforts to promote democracy worldwide, or that had a significant impact on global affairs."
    background = "I am a high school student writing a report for an assignment about influential women in politics that will focus on Condoleezza Rice."
    report_type = "SCALE25_report2"

    results_df = asyncio.run(
        research_prompt_report_eval(
            query=query,
            background=background,
            report_type=report_type,
            num_to_generate=1
        )
    )
    results_df.to_csv("reports_with_argue_eval3.csv", index=False)
    print("Saved results to reports_with_argue_eval3.csv")

if __name__ == "__main__":
    main()