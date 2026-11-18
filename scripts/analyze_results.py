import json
from pathlib import Path


def main() -> None:
    data_file = Path("docs") / "promt-experiment" / "prompt_experiments_data.json"
    if not data_file.is_file():
        data_file = Path("docs") / "prompt_experiments_data.json"

    if not data_file.is_file():
        print(f"Error: prompt experiments data file not found at {data_file}")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print("AGGREGATE SUMMARY BY PROMPT VERSION")
    print("=" * 60)

    for ver in ["V1", "V2", "V3"]:
        subset = [d for d in data if d["version"] == ver]
        if not subset:
            continue
        valid_json = sum(1 for d in subset if d["is_valid_json"])
        avg_in = sum(d["input_tokens"] for d in subset) / len(subset)
        avg_out = sum(d["output_tokens"] for d in subset) / len(subset)
        avg_lat = sum(d["latency_ms"] for d in subset) / len(subset)
        avg_cost = sum(d["estimated_cost_usd"] for d in subset) / len(subset)
        print(f"Version {ver}:")
        print(f"  Strict JSON Valid: {valid_json}/{len(subset)} ({valid_json * 10}%)")
        print(f"  Avg Input Tokens:  {avg_in:.1f}")
        print(f"  Avg Output Tokens: {avg_out:.1f}")
        print(f"  Avg Latency:       {avg_lat:.1f} ms")
        print(f"  Avg Cost per Call: ${avg_cost:.7f} USD")
        print()

    print("=" * 60)
    print("PER-TICKET COMPARISON (Category & Urgency & Churn)")
    print("=" * 60)

    tickets = sorted(list(set(d["ticket_id"] for d in data)))
    for tid in tickets:
        v1_entry = next((d for d in data if d["ticket_id"] == tid and d["version"] == "V1"), None)
        v2_entry = next((d for d in data if d["ticket_id"] == tid and d["version"] == "V2"), None)
        v3_entry = next((d for d in data if d["ticket_id"] == tid and d["version"] == "V3"), None)

        if not v1_entry or not v2_entry or not v3_entry:
            continue

        print(f"\n[{tid}] {v1_entry['ticket_title']}")
        # V1 snippet
        v1_preview = v1_entry["raw_reply"][:150].replace("\n", " ")
        print(f"  V1: {v1_preview}...")

        # V2 parsed
        if v2_entry["is_valid_json"] and v2_entry["parsed_json"]:
            v2_p = v2_entry["parsed_json"]
            print(f"  V2: Category='{v2_p.get('category')}' | Urgency='{v2_p.get('urgency')}' | Churn={v2_p.get('churn_risk')} | Sentiment='{v2_p.get('sentiment')}'")
            print(f"      Summary: {v2_p.get('summary')}")
            print(f"      Entities: {v2_p.get('key_entities')}")
        else:
            print(f"  V2: Non-JSON raw: {v2_entry['raw_reply'][:100]}")

        # V3 parsed
        if v3_entry["is_valid_json"] and v3_entry["parsed_json"]:
            v3_p = v3_entry["parsed_json"]
            print(f"  V3: Category='{v3_p.get('category')}' | Urgency='{v3_p.get('urgency')}' | Churn={v3_p.get('churn_risk')} | Sentiment='{v3_p.get('sentiment')}'")
            print(f"      Summary: {v3_p.get('summary')}")
            print(f"      Entities: {v3_p.get('key_entities')}")
        else:
            print(f"  V3: Non-JSON raw: {v3_entry['raw_reply'][:100]}")


if __name__ == "__main__":
    main()
