"""
Example: Load the 12-prompt sample dataset without errors, then save to SQLite.
Run from usecase_1: python example_load_sample.py
"""

from pathlib import Path

from data_layer import PromptStore

def main() -> None:
    sample_path = Path(__file__).parent / "sample_prompts.json"
    store = PromptStore()
    records = store.load_prompts(str(sample_path))
    print(f"Loaded {len(records)} prompts without errors.")
    for r in records[:3]:
        print(f"  - {r.prompt_id} ({r.layer}): {r.name}")
    if len(records) > 3:
        print(f"  ... and {len(records) - 3} more.")

    db_path = Path(__file__).parent / "prompts.db"
    store.save_prompts(records, str(db_path))
    print(f"Saved to {db_path}.")

    # Reload from DB to verify
    store2 = PromptStore()
    store2.open_db(str(db_path))
    assert len(store2.get_all_prompts()) == 12
    print("Verified: reload from SQLite returns 12 prompts.")

if __name__ == "__main__":
    main()
