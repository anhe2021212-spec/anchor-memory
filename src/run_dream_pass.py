#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cron trigger for the anchor-sse-owned nightly weak-flow maintenance."""
import os
import json
import urllib.request
from datetime import datetime

def main():
    endpoint = os.environ.get(
        "ANCHOR_NIGHT_REPAIR_URL",
        "http://127.0.0.1:8765/api/internal/night-flow-repair",
    )
    request = urllib.request.Request(
        endpoint,
        data=json.dumps({"node_limit": 80, "edge_limit": 80}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        stats = json.loads(response.read().decode("utf-8"))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] dream_pass: {json.dumps(stats, ensure_ascii=False)}"
    print(line)

if __name__ == "__main__":
    main()
