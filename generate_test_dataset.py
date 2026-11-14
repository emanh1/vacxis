"""Generate a golden evaluation dataset for the Vaccine RAG system.

Thin CLI shim around :mod:`vacxis_capstone_fa26.testdata.cli` (needs a GPU
host with vLLM; see the module docstring for the full flag list). Run with::

    uv run generate_test_dataset.py
    uv run generate_test_dataset.py --vllm-model /path/to/Qwen --checkpoint ckpt.json
    uv run generate_test_dataset.py --limit 10 --out smoke.json

Output is two files: the unmasked head and the symptom-masked tail.
"""
from __future__ import annotations

from vacxis_capstone_fa26.testdata.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
