from argue_eval.judge import evaluate_report
from pathlib import Path

async def run_evaluation(report, nuggets_path, provider="hltcoe_local"):
    """
    Evaluate report using Argue-Eval.
    """
    return await evaluate_report(report, Path(nuggets_path), provider)