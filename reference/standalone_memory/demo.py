"""
standalone_memory/demo.py
Demonstration script showing the 4 tiers of SAGE Memory in action.
Run with: python -m standalone_memory.demo
"""

import json
from pathlib import Path
import sys

# Ensure parent directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from standalone_memory import (
    memory_store_hot,
    memory_store_cold,
    memory_promote_to_cold,
    memory_search_hot,
    memory_search_cold,
    memory_get_context,
    memory_engine,
)

def main():
    print("=" * 60)
    print("  SAGE STANDALONE MULTI-TIER MEMORY ENGINE DEMO")
    print("=" * 60)

    user_id = "developer_alex"
    chat_id = "session_001"

    # 1. Store Global Directives
    print("\n[1] Storing Global Directive...")
    r1 = memory_store_cold(
        user_id=user_id,
        content="Always write production code in Python with strict type hints.",
        category="instruction",
        is_global=True,
    )
    print(" -> Global stored:", r1["memory"]["memory_id"])

    # 2. Store Session Working Memory (Hot)
    print("\n[2] Storing Hot Working Memory (tied to current session)...")
    r2 = memory_store_hot(
        user_id=user_id,
        chat_id=chat_id,
        content="User is debugging Docker port binding on port 8080.",
        category="task",
        importance=0.9,
    )
    hot_id = r2["memory"]["memory_id"]
    print(f" -> Hot memory stored: {hot_id} (content: '{r2['memory']['content']}')")

    # 3. Store Permanent Knowledge (Cold)
    print("\n[3] Storing Cold Long-Term Knowledge...")
    r3 = memory_store_cold(
        user_id=user_id,
        content="Database credentials and cluster endpoints are managed via Vault.",
        category="technical",
        importance=0.8,
    )
    print(" -> Cold memory stored:", r3["memory"]["memory_id"])

    # 4. Search Hot Tier
    print("\n[4] Performing Semantic Search on Hot Tier...")
    search_hot = memory_search_hot(query="What port is being debugged?", user_id=user_id)
    print(" -> Hot Search Matches:", len(search_hot["memories"]))
    for m in search_hot["memories"]:
        print(f"    • [{m['category']}] {m['content']} (similarity: {m['similarity']})")

    # 5. Promote Hot Memory to Cold
    print(f"\n[5] Promoting Hot Memory '{hot_id}' to Cold Tier...")
    prom = memory_promote_to_cold(memory_id=hot_id)
    print(" -> Promotion result:", prom)

    # 6. Add Chat Turns to Ledger
    print("\n[6] Logging Conversation Turns into Ledger...")
    memory_engine.add_message(chat_id=chat_id, role="user", content="Hey SAGE, I am having trouble with Docker port mapping.", user_id=user_id)
    memory_engine.add_message(chat_id=chat_id, role="assistant", content="Check if another container is binding 8080.", user_id=user_id)
    memory_engine.add_message(chat_id=chat_id, role="user", content="Good catch, killed the old container. What language should I write the test in?", user_id=user_id)
    print(" -> 3 messages recorded.")

    # 7. Assemble Complete Model Context (The Agent Integration Call)
    print("\n[7] Calling memory_get_context() to assemble prompt context...")
    context_res = memory_get_context(
        query="What language should I write the test in?",
        chat_id=chat_id,
        user_id=user_id,
    )
    print("\n" + "-" * 50)
    print(context_res["context_text"])
    print("-" * 50)
    print(f"Estimated Tokens: {context_res['token_estimate']}")
    print(f"Context Breakdown: {json.dumps(context_res['breakdown'], indent=2)}")

    # 8. Inspect Activity Audit Trail
    print("\n[8] Recent Activity Audit Log:")
    activities = memory_engine.get_activity_log(limit=5)
    for act in activities:
        print(f"  • [{act['timestamp'][:19]}] {act['action'].upper():12} | Tier: {str(act['tier']):4} | {act['details']}")

    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
