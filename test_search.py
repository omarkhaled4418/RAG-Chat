import sys
from app.config import Config
from app.rag_engine import RAGEngine

engine = RAGEngine(Config)
query = "What is this document about?"
print(f"Query: {query}")

results = engine.store.search(query, top_k=3)
print(f"Search results count: {len(results)}")
for r in results:
    print(f"Score: {r['score']} - {r['text'][:100]}...")

prompt = engine.ask(query)
print("\nLLM Answer:")
print(prompt['answer'])
