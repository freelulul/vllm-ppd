#!/usr/bin/env python3
"""
Download WildChat-1M and LMSYS-Chat-1M datasets for PPD testing.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

def json_serial(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

DATA_DIR = Path(PROJECT_DIR) / "data"
DATA_DIR.mkdir(exist_ok=True)

def download_wildchat():
    """Download WildChat-1M dataset and save as JSON."""
    print("\n" + "="*60)
    print("Downloading WildChat-1M...")
    print("="*60)

    from datasets import load_dataset

    start = time.time()
    try:
        # Load dataset
        dataset = load_dataset("allenai/WildChat-1M")

        elapsed = time.time() - start
        print(f"\nWildChat-1M downloaded in {elapsed:.1f}s")
        print(f"  Train split: {len(dataset['train'])} rows")

        # Save full dataset as JSON
        output_path = DATA_DIR / "WildChat_1M.json"
        print(f"\nConverting to JSON format...")

        conversations = []
        for i, item in enumerate(dataset['train']):
            conv = {
                "id": item.get("conversation_hash", f"wildchat_{i}"),
                "conversations": item.get("conversation", []),
                "model": item.get("model", "unknown"),
                "turn_count": item.get("turn", 1),
                "language": item.get("language", "unknown"),
            }
            conversations.append(conv)
            if (i + 1) % 100000 == 0:
                print(f"  Processed {i + 1} conversations...")

        print(f"Saving to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, default=json_serial)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Full dataset saved: {output_path} ({size_mb:.1f} MB)")

        # Count multi-turn
        multi_turn = sum(1 for c in conversations if c["turn_count"] >= 2)
        print(f"  Multi-turn conversations: {multi_turn} ({100*multi_turn/len(conversations):.1f}%)")

        return True
    except Exception as e:
        print(f"ERROR downloading WildChat-1M: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_lmsys():
    """Download LMSYS-Chat-1M dataset."""
    print("\n" + "="*60)
    print("Downloading LMSYS-Chat-1M...")
    print("="*60)

    from datasets import load_dataset

    start = time.time()
    try:
        # Load dataset (will be cached locally)
        dataset = load_dataset("lmsys/lmsys-chat-1m")

        elapsed = time.time() - start
        print(f"\nLMSYS-Chat-1M downloaded successfully in {elapsed:.1f}s")
        print(f"  Train split: {len(dataset['train'])} rows")

        # Save sample for quick analysis
        sample = dataset['train'].select(range(min(1000, len(dataset['train']))))
        sample.to_json(DATA_DIR / "lmsys_sample_1k.json")
        print(f"  Sample saved to: {DATA_DIR / 'lmsys_sample_1k.json'}")

        return True
    except Exception as e:
        print(f"ERROR downloading LMSYS-Chat-1M: {e}")
        return False


def main():
    print("="*60)
    print("Dataset Download Script")
    print("="*60)
    print(f"Data directory: {DATA_DIR}")

    # Download WildChat only (LMSYS is gated)
    wildchat_ok = download_wildchat()

    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    print(f"  WildChat-1M: {'OK' if wildchat_ok else 'FAILED'}")
    print(f"  LMSYS-Chat-1M: SKIPPED (gated dataset, requires HuggingFace approval)")

    if wildchat_ok:
        print("\nWildChat downloaded successfully!")
        print(f"  Output: {DATA_DIR / 'WildChat_1M.json'}")
    else:
        print("\nDownload failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
