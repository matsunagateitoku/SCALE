import json
from collections import defaultdict
import os

def flatten_facts(data):
    required_facts = []

    for q, info in data.items():
        if isinstance(info, list) and info[0] == "OR":
            or_facts = info[1]
            required_facts.append(("OR", or_facts))
        elif isinstance(info, list):
            for item in info:
                if isinstance(item, dict):
                    required_facts.append(("AND", item))
        elif isinstance(info, dict):
            required_facts.append(("AND", info))
        else:
            print(f"Unknown format for question: {q}")
    return required_facts

def select_min_doc_ids(required_facts):
    all_facts = []

    for kind, fact_dict in required_facts:
        if kind == "OR":
            best_fact = min(fact_dict.items(), key=lambda x: len(x[1]) if x[1] else float('inf'))
            all_facts.append(best_fact)
        else:
            all_facts.extend(fact_dict.items())

    fact_to_docs = {fact: set(docs) for fact, docs in all_facts if docs}
    uncovered_facts = set(fact_to_docs.keys())
    selected_docs = set()

    while uncovered_facts:
        doc_to_facts = defaultdict(set)
        for fact in uncovered_facts:
            for doc in fact_to_docs[fact]:
                doc_to_facts[doc].add(fact)

        best_doc = max(doc_to_facts.items(), key=lambda x: len(x[1]))[0]
        selected_docs.add(best_doc)
        uncovered_facts -= doc_to_facts[best_doc]

    return selected_docs

# === MAIN ===
if __name__ == "__main__":
    base_dir = "/exp/scale25/neuclir/eval/nuggets/individual_lang"
    langs = ['zho', 'fas', 'rus']
    #ids = [324, 361, 387]
    ids = [300, 303, 308, 309, 310, 334, 335, 343, 351, 352, 365, 367, 372, 373, 377, 380, 382, 383, 388
]
    

    for lang in langs:
        for req_id in ids:
            file_path = os.path.join(base_dir, f"nuggets_{lang}_{req_id}.json")
            print(f"\n📂 Processing {file_path}")

            if not os.path.isfile(file_path):
                print(f"❌ File not found: {file_path}")
                continue

            with open(file_path) as f:
                data = json.load(f)

            required_facts = flatten_facts(data)
            minimal_doc_ids = select_min_doc_ids(required_facts)

            print(f"✅ Minimal document IDs for {lang}-{req_id}:")
            for doc_id in sorted(minimal_doc_ids):
                print(doc_id)
