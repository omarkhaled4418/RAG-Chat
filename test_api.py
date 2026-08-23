"""Test suite for headless RAG Chat API."""

import json
from run import create_app

app = create_app()
client = app.test_client()

print("1. Testing GET / (Root Health & API Catalog)...")
r1 = client.get("/")
assert r1.status_code == 200
data1 = r1.get_json()
print("   Status:", data1["status"])
print("   Endpoints:", list(data1["endpoints"].keys()))

print("\n2. Testing POST /api/documents/text (Direct text ingestion)...")
r2 = client.post(
    "/api/documents/text",
    json={
        "text": "Antigravity Superchargers are located in Sector 7 and provide 350kW ultra-fast charging.",
        "title": "chargers_guide.txt",
    },
)
assert r2.status_code == 200
print("   Ingested:", r2.get_json())

print("\n3. Testing GET /api/documents (Document list)...")
r3 = client.get("/api/documents")
assert r3.status_code == 200
print("   Documents:", r3.get_json())

print("\n4. Testing POST /api/search (Fast similarity search)...")
r4 = client.post("/api/search", json={"query": "Where is the 350kW supercharger?", "top_k": 3})
assert r4.status_code == 200
search_data = r4.get_json()
print(f"   Matches found: {search_data['count']} in {search_data['latency_ms']}ms")
for item in search_data["results"]:
    print(f"   - Match: {item['text']} (source: {item['source']}, score: {item['score']})")

print("\n5. Testing GET /api/index/status...")
r5 = client.get("/api/index/status")
assert r5.status_code == 200
print("   Index status:", r5.get_json())

print("\n6. Testing GET /api/history...")
r6 = client.get("/api/history")
assert r6.status_code == 200
print("   History status:", r6.get_json())

print("\n7. Testing DELETE /api/documents/<filename>...")
r7 = client.delete("/api/documents/chargers_guide.txt")
assert r7.status_code == 200
print("   Delete status:", r7.get_json())

print("\n8. Testing POST /api/index/clear...")
r8 = client.post("/api/index/clear")
assert r8.status_code == 200
print("   Clear status:", r8.get_json())

print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")
