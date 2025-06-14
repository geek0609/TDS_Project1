import json

# Check knowledge base structure
print("=== KNOWLEDGE BASE STRUCTURE ===")
with open('scripts/processed/knowledge_base.json', 'r', encoding='utf-8') as f:
    kb = json.load(f)

print(f"Number of topics: {len(kb['topics'])}")
print(f"Sample topic keys: {list(kb['topics'][0].keys())}")
print(f"Sample topic: {kb['topics'][0]}")

print("\n=== Q&A STRUCTURE ===")
with open('scripts/processed/qa_pairs.json', 'r', encoding='utf-8') as f:
    qa = json.load(f)

print(f"Number of Q&A pairs: {len(qa['qa_pairs'])}")
print(f"Sample Q&A keys: {list(qa['qa_pairs'][0].keys())}")
print(f"Sample Q&A: {qa['qa_pairs'][0]}") 