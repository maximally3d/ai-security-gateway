"""
Data preparation script for AI Security Gateway training.
"""

import json
import random
from pathlib import Path
from typing import Dict, List


def load_raw_queries(file_path: str) -> List[Dict]:
    queries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append({"input": line})
    return queries


def classify_query(query: str) -> Dict[str, str]:
    high_risk_keywords = [
        "password", "api_key", "api-key", "secret",
        "transaction", "client", "database", "sql", "select",
        "quantumpay", "crystal", "ariadne", "sertech", "alfa-bank",
        "salary", "source code", "encryption algorithm"
    ]

    medium_risk_keywords = [
        "api", "integration", "internal",
        "module", "project",
        "report", "document"
    ]

    query_lower = query.lower()

    for keyword in high_risk_keywords:
        if keyword in query_lower:
            return {
                "risk_level": "HIGH",
                "reason": f"Detected mention of confidential information: '{keyword}'"
            }

    for keyword in medium_risk_keywords:
        if keyword in query_lower:
            return {
                "risk_level": "MEDIUM",
                "reason": f"Indirect mention of internal processes: '{keyword}'"
            }

    return {
        "risk_level": "LOW",
        "reason": "Query does not contain obvious signs of confidential information"
    }


def format_for_training(query: str, classification: Dict[str, str]) -> Dict:
    output = json.dumps(classification, ensure_ascii=False)
    return {
        "instruction": "Analyze query for confidential information.",
        "input": query,
        "output": output
    }


def balance_dataset(examples: List[Dict]) -> List[Dict]:
    high = [e for e in examples if json.loads(e["output"])["risk_level"] == "HIGH"]
    medium = [e for e in examples if json.loads(e["output"])["risk_level"] == "MEDIUM"]
    low = [e for e in examples if json.loads(e["output"])["risk_level"] == "LOW"]

    target_size = max(len(high), len(medium), len(low))

    def oversample(items, target):
        if len(items) >= target:
            return items
        result = items.copy()
        while len(result) < target:
            result.append(random.choice(items))
        random.shuffle(result)
        return result

    balanced = []
    balanced.extend(oversample(high, target_size))
    balanced.extend(oversample(medium, target_size))
    balanced.extend(oversample(low, target_size))
    random.shuffle(balanced)
    return balanced


def main():
    print("=" * 60)
    print("AI Security Gateway - Data Preparation")
    print("=" * 60)

    input_dir = Path(__file__).parent.parent / "examples"
    output_path = input_dir / "sample_queries.jsonl"
    raw_queries_path = input_dir / "raw_queries.txt"

    if raw_queries_path.exists():
        print(f"\nLoading raw queries from: {raw_queries_path}")
        raw_queries = load_raw_queries(str(raw_queries_path))
        print(f"Loaded {len(raw_queries)} queries")

        print("\nClassifying queries...")
        examples = []
        for q in raw_queries:
            classification = classify_query(q["input"])
            example = format_for_training(q["input"], classification)
            examples.append(example)

        print("Balancing dataset...")
        balanced_examples = balance_dataset(examples)

        print(f"\nSaving to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            for example in balanced_examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        print(f"Saved {len(balanced_examples)} examples")

        high_count = sum(1 for e in balanced_examples if json.loads(e["output"])["risk_level"] == "HIGH")
        medium_count = sum(1 for e in balanced_examples if json.loads(e["output"])["risk_level"] == "MEDIUM")
        low_count = sum(1 for e in balanced_examples if json.loads(e["output"])["risk_level"] == "LOW")

        print(f"\nDataset statistics:")
        print(f"  HIGH:   {high_count}")
        print(f"  MEDIUM: {medium_count}")
        print(f"  LOW:    {low_count}")
    else:
        print(f"\nNo raw queries file found at: {raw_queries_path}")
        print("Using existing sample_queries.jsonl")
        print("To prepare custom data, create examples/raw_queries.txt with one query per line")


if __name__ == "__main__":
    main()
