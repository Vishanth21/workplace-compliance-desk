"""Verification script for compliance advisory API endpoints.

Validates the 5 core checks:
1. Health: GET /health -> {"status":"ok"}
2. Submit: POST /runs with sample.json -> 202 with run_id
3. Poll: GET /runs/<id> -> succeeded, one citation
4. Reject: POST with empty question and headcount 0 -> 400 with two field errors
5. Metrics: GET /metrics -> plaintext metrics
"""

import json
import time
import httpx

BASE_URL = "http://127.0.0.1:8000"


def run_checks():
    client = httpx.Client(base_url=BASE_URL, timeout=10.0)

    print("\n--- Check 1: Health check ---")
    r1 = client.get("/health")
    print(f"Status: {r1.status_code}, Body: {r1.text}")
    assert r1.status_code == 200
    assert r1.json() == {"status": "ok"}
    print("PASS: Health check ok")

    print("\n--- Check 2: Submit valid request (sample.json) ---")
    with open("sample.json", "r") as f:
        payload = json.load(f)
    r2 = client.post("/runs", json=payload)
    print(f"Status: {r2.status_code}, Body: {r2.text}")
    assert r2.status_code == 202
    data2 = r2.json()
    run_id = data2["run_id"]
    correlation_id = data2["correlation_id"]
    status_url = data2["status_url"]
    print(f"PASS: Run accepted with run_id={run_id}, correlation_id={correlation_id}")

    print("\n--- Check 3: Poll until succeeded with citation ---")
    poll_start = time.time()
    succeeded = False
    for attempt in range(10):
        r3 = client.get(status_url)
        data3 = r3.json()
        print(f"Poll #{attempt + 1} ({time.time() - poll_start:.1f}s): status={data3.get('status')}")
        if data3.get("status") == "succeeded":
            succeeded = True
            print(f"Advisory: {data3.get('answer')}")
            print(f"Citations: {data3.get('citations')}")
            assert data3["route"] == "answer-with-citation"
            assert len(data3["citations"]) == 1
            assert data3["citations"][0]["chunk_id"] == "css2020-s53-1-prov-2"
            break
        time.sleep(0.5)

    assert succeeded, "Run did not succeed within expected time!"
    print("PASS: Poll succeeded with cited provision")

    print("\n--- Check 4: Ingestion rejection & quarantine ---")
    malformed_payload = {
        "question": "",
        "establishment": {
            "headcount": 0,
            "industry": "Software",
            "state": "Karnataka",
        },
    }
    r4 = client.post("/runs", json=malformed_payload)
    print(f"Status: {r4.status_code}, Body: {r4.text}")
    assert r4.status_code == 400
    data4 = r4.json()
    assert data4["status"] == "quarantined"
    assert len(data4["errors"]) == 2
    fields = [err["field"] for err in data4["errors"]]
    print(f"Quarantine field errors: {fields}")
    assert "question" in fields
    assert "establishment.headcount" in fields
    print("PASS: Ingestion rejection and quarantine verified with exactly two field errors")

    print("\n--- Check 5: Plaintext metrics ---")
    r5 = client.get("/metrics")
    print(f"Status: {r5.status_code}")
    print(r5.text)
    assert r5.status_code == 200
    assert "compliance_runs_total" in r5.text
    assert "compliance_queue_depth" in r5.text
    print("PASS: Metrics endpoint verified with queue depth")

    print("\n==========================================")
    print("ALL API VERIFICATION CHECKS PASSED!")
    print("==========================================\n")


if __name__ == "__main__":
    run_checks()
