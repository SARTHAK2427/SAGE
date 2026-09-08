"""
view_chroma.py - Quick viewer for SAGE ChromaDB collections and records.
Run with: python view_chroma.py
"""

import sys
from pathlib import Path
import chromadb

# Resolve path
chroma_dir = Path(__file__).resolve().parent / "chroma_db"
if not chroma_dir.exists():
    print(f"Error: {chroma_dir} does not exist.")
    sys.exit(1)

client = chromadb.PersistentClient(path=str(chroma_dir))
collections = client.list_collections()

print("\n" + "=" * 60)
print(f"  SAGE CHROMADB INSPECTOR")
print(f"  Directory : {chroma_dir}")
print(f"  Version   : {chromadb.__version__}")
print(f"  Total Collections: {len(collections)}")
print("=" * 60)

if not collections:
    print("\nNo collections found.")
    sys.exit(0)

for col in collections:
    count = col.count()
    print(f"\n[Collection: {col.name}] -> {count} record(s)")
    print("-" * 50)
    
    if count == 0:
        print("  (Empty collection)")
        continue
    
    data = col.get(include=["documents", "metadatas"])
    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    for i in range(len(ids)):
        print(f"  [{i + 1}] ID       : {ids[i]}")
        print(f"      Document : {docs[i]}")
        print(f"      Metadata : {metas[i]}")

print("\n" + "=" * 60 + "\n")
