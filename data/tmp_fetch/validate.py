#!/usr/bin/env python3
"""Validate fetched CMS JSON page files: parse, count records, list provider types."""

import json
import sys

for path in sys.argv[1:]:
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"{path}: FAIL not-a-list ({type(data).__name__})")
            continue
        types = sorted({r.get("Rndrng_Prvdr_Type", "?") for r in data})
        keys_ok = all("Rndrng_NPI" in r and "HCPCS_Cd" in r for r in data)
        print(f"{path}: OK {len(data)} records, keys_ok={keys_ok}, types={types}")
    except Exception as e:
        print(f"{path}: FAIL {e.__class__.__name__}: {e}")
