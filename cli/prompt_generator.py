import pandas as pd
import itertools

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

def generate_prompt_df(query, background, instructions):
    """
    Generate a DataFrame containing all prompt variants.
    """
    all_combinations = list(itertools.product(
        instructions["total_words_options"],
        instructions["focus_instructions"],
        instructions["structure_instructions"],
        instructions["fact_instructions"],
        instructions["length_instructions"],
        instructions["depth_instructions"],
        instructions["opinion_instructions"],
        instructions["bias_instructions"],
        instructions["citation_instructions"],
        instructions["tone_instructions"],
        instructions["formatting_instructions"],
        instructions["extra_instructions"]
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
    return pd.DataFrame(prompt_variants)