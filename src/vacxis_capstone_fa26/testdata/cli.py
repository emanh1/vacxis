"""CLI entry point for the test data generator.

Two sources are supported:

* ``--source queries`` (default): convert raw user queries from
  ``query.parquet`` into grounded vaccine-domain queries. Rewriting is done
  by Qwen2.5 via vLLM (``--vllm``) or by an offline rule-based rewriter
  (``--no-vllm``). Reference answers follow the same flag.
* ``--source templates``: the legacy template/paraphrase generator.

Optional post-stages (vLLM path): ``--mask-symptoms`` rewrites the query
symptoms of the records past ``--split-at`` to ones the evidence discusses;
``--relabel`` re-derives the answerability label, the reference answer and
the claim map from the *final* query, and drops records whose citations do
not resolve or whose evidence quotes are not verbatim.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

from .answer_generator import (
    ExtractiveAnswerGenerator,
    NoOpAnswerGenerator,
    OpenRouterAnswerGenerator,
)
from .chunks import index_by_category, index_by_vaccine, load_chunks
from .generator import (
    RuleBasedRewriter,
    derive_default_seed,
    generate_and_validate,
    generate_and_validate_from_source,
)
from .validator import validate
from .postprocess import apply_replacements, filter_no_han, split_records
from .relabel import apply_verification, verify_record
from .vllm_rewrite import (
    DEFAULT_VLLM_MODEL,
    QwenAnswerGenerator,
    QwenLabelVerifier,
    QwenRewriter,
    QwenSymptomMasker,
    VLLMCheckpoint,
    VLLMClient,
    VLLMError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "chunks_storage" / "chunks.parquet"
DEFAULT_MAPPING_PATH = PROJECT_ROOT / "75_vaccine_names_mapping.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "test_dataset.json"
DEFAULT_SOURCE_QUERIES = PROJECT_ROOT / "scrape_dataset" / "query.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a golden evaluation dataset grounded on chunks.parquet",
    )
    p.add_argument(
        "--source",
        choices=["queries", "templates"],
        default="queries",
        help=(
            "Query source: 'queries' converts query.parquet (default); "
            "'templates' uses the legacy template generator."
        ),
    )
    p.add_argument(
        "--source-queries",
        type=Path,
        default=DEFAULT_SOURCE_QUERIES,
        help="Path to the source query.parquet (id, query).",
    )
    p.add_argument(
        "--vllm",
        action="store_true",
        default=None,
        help="Use Qwen2.5 via vLLM for rewriting and answers (GPU host).",
    )
    p.add_argument(
        "--no-vllm",
        action="store_true",
        help="Use the offline rule-based rewriter and extractive answers.",
    )
    p.add_argument(
        "--vllm-model",
        type=str,
        default=DEFAULT_VLLM_MODEL,
        help="vLLM model id or local path (default: %(default)s).",
    )
    p.add_argument(
        "--vllm-gpu-memory",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization (default: %(default)s).",
    )
    p.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks.parquet (default: chunks_storage/chunks.parquet)",
    )
    p.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING_PATH,
        help="Path to vaccine name mapping JSON (id -> name)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON file (default: test_dataset.json at project root)",
    )
    p.add_argument(
        "--per-type",
        type=int,
        default=0,
        help=(
            "queries source: cap on queries per QUERY_TYPE (0 = no cap, all). "
            "templates source: queries per type (0 falls back to the legacy 3)."
        ),
    )
    p.add_argument(
        "--distractors",
        type=int,
        default=2,
        help="Number of distractor chunks to attach per query (default: 2)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for reproducibility. If omitted, a deterministic "
            "seed is derived from the chunk content so re-runs are stable."
        ),
    )
    p.add_argument(
        "--generate-answers",
        action="store_true",
        help=(
            "Generate REFERENCE_ANSWER/CLAIM_EVIDENCE_MAP via OpenRouter "
            "(templates source only; the queries source uses vLLM/extractive)."
        ),
    )
    p.add_argument(
        "--answers-model",
        type=str,
        default=None,
        help="OpenRouter model override for reference answers",
    )
    p.add_argument(
        "--dry-run-answers",
        action="store_true",
        help="Force offline extractive answers regardless of --vllm.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "queries source only: process at most the first N source queries "
            "(0 = all). Useful for quick smoke runs before a full pass."
        ),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "vLLM source only: JSON file to cache rewrites + answers so a "
            "crashed/killed run can resume. Re-run with the same path to "
            "continue; the meta header rejects a model/seed/prompt mismatch."
        ),
    )
    p.add_argument(
        "--answers-batch-size",
        type=int,
        default=64,
        help="vLLM source only: answer-generation batch size (default: %(default)s).",
    )
    p.add_argument(
        "--rewrite-batch-size",
        type=int,
        default=256,
        help=(
            "vLLM source only: rewrite batch size (default: %(default)s). "
            "Each batch is one LLM call and one checkpoint save point."
        ),
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the post-write validation pass.",
    )
    p.add_argument(
        "--strict-validation",
        action="store_true",
        help="Treat validation warnings as failures (exit non-zero).",
    )
    p.add_argument(
        "--filter-no-han",
        action="store_true",
        help="Drop records containing any Han (CJK) character before writing.",
    )
    p.add_argument(
        "--relabel",
        action="store_true",
        help=(
            "Re-derive ANSWERABILITY, REFERENCE_ANSWER and CLAIM_EVIDENCE_MAP "
            "from the final query and its evidence, then drop records that do "
            "not satisfy the invariants. Needs the vLLM path."
        ),
    )
    p.add_argument(
        "--relabel-scope",
        choices=["all", "masked"],
        default="all",
        help=(
            "Which records --relabel covers: 'all' (default) or only the "
            "symptom-masked tail. Only used with --relabel."
        ),
    )
    p.add_argument(
        "--verify-batch-size",
        type=int,
        default=64,
        help="vLLM source only: verification batch size (default: %(default)s).",
    )
    p.add_argument(
        "--mask-symptoms",
        action="store_true",
        help=(
            "For records after the first --split-at, replace query symptoms "
            "with symptoms found in the record's most relevant chunk."
        ),
    )
    p.add_argument(
        "--split-at",
        type=int,
        default=300,
        help=(
            "Boundary between the normal export (first N records) and the "
            "symptom-masked remainder (default: %(default)s). Only used with "
            "--mask-symptoms."
        ),
    )
    p.add_argument(
        "--out-masked",
        type=Path,
        default=None,
        help=(
            "Where to write the symptom-masked remainder (default: <out> "
            "with a _masked suffix)."
        ),
    )
    return p.parse_args(argv)


def _resolve_use_vllm(args: argparse.Namespace) -> bool:
    if args.no_vllm:
        return False
    if args.vllm:
        return True
    return args.source == "queries"


def _load_source_queries(path: Path) -> list[dict]:
    table = pq.read_table(str(path), columns=["id", "query"])
    rows = table.to_pylist()
    return [{"id": r["id"], "query": r["query"]} for r in rows]


def _cap_by_type(queries: list[dict], per_type: int) -> list[dict]:
    """Keep at most ``per_type`` queries per QUERY_TYPE (stable by QUERY_ID)."""
    if per_type <= 0:
        return queries
    counts: dict[str, int] = {}
    kept: list[dict] = []
    for q in sorted(queries, key=lambda r: r.get("QUERY_ID", "")):
        t = q.get("QUERY_TYPE", "")
        counts[t] = counts.get(t, 0) + 1
        if counts[t] <= per_type:
            kept.append(q)
    kept.sort(key=lambda r: r.get("QUERY_ID", ""))
    return kept


def _run_queries_source(
    args: argparse.Namespace,
    chunks,
    rng: random.Random,
    vaccine_mapping: dict[str, str],
    seed: int,
    checkpoint: VLLMCheckpoint | None,
) -> tuple[list[dict], object]:
    source_queries = _load_source_queries(args.source_queries)
    print(f"Loaded {len(source_queries)} source queries from {args.source_queries}")
    if args.limit and args.limit < len(source_queries):
        source_queries = source_queries[: args.limit]
        print(f"  -> --limit applied: processing first {len(source_queries)}")

    use_vllm = _resolve_use_vllm(args)
    vaccine_names = list(index_by_vaccine(chunks).keys())

    if use_vllm:
        client = VLLMClient(
            model=args.vllm_model,
            gpu_memory_utilization=args.vllm_gpu_memory,
            seed=seed,
        )
        rewriter = QwenRewriter(
            client,
            vaccine_names=vaccine_names,
            checkpoint=checkpoint,
            rewrite_batch_size=args.rewrite_batch_size,
        )
        if args.dry_run_answers:
            ans_gen = ExtractiveAnswerGenerator()
        else:
            ans_gen = QwenAnswerGenerator(
                client,
                checkpoint=checkpoint,
                batch_size=args.answers_batch_size,
            )
        print(f"Using Qwen via vLLM: {args.vllm_model}")
    else:
        rewriter = RuleBasedRewriter(chunks, rng)
        ans_gen = ExtractiveAnswerGenerator()
        print("Using offline rule-based rewriter + extractive answers.")

    queries, rpt = generate_and_validate_from_source(
        chunks,
        source_queries,
        rewriter,
        rng,
        args.distractors,
        vaccine_mapping=vaccine_mapping,
        answer_generator=ans_gen,
    )
    if args.per_type > 0:
        queries = _cap_by_type(queries, args.per_type)
        rpt = validate(
            queries,
            chunks,
            index_by_category(chunks),
            expected_per_type=0,
            expected_types=sorted({q.get("QUERY_TYPE") for q in queries}),
        )
    return queries, rpt


def _build_masker(
    args: argparse.Namespace,
    chunks,
    rng: random.Random,
    checkpoint: VLLMCheckpoint | None,
    seed: int,
):
    """Build the symptom masker for the masked-tail export.

    Uses the vLLM model when available so the model picks the symptoms to
    replace (it sees the query and the score-3 chunks). Falls back to the
    offline rule-based path, which cannot replace symptoms -- there the
    records are exported unchanged.
    """
    if _resolve_use_vllm(args):
        client = VLLMClient(
            model=args.vllm_model,
            gpu_memory_utilization=args.vllm_gpu_memory,
            seed=seed,
        )
        print(f"Masking symptoms with Qwen via vLLM: {args.vllm_model}")
        return QwenSymptomMasker(
            client,
            checkpoint=checkpoint,
            batch_size=args.answers_batch_size,
        )

    print("vLLM unavailable: the masked export will keep its queries unchanged.")
    return _NoopMasker()


class _NoopMasker:
    """Offline masker that proposes no replacements."""

    def mask(self, records: list[dict]) -> list[list[dict[str, str]]]:
        return [[] for _ in records]


def _build_verifier(
    args: argparse.Namespace,
    chunks,
    rng: random.Random,
    checkpoint: VLLMCheckpoint | None,
    seed: int,
):
    """Build the label/answer verifier for the ``--relabel`` stage.

    Needs the vLLM path: without it there is no model to re-derive the
    label and the answer from the final query.
    """
    if not _resolve_use_vllm(args):
        raise VLLMError("--relabel needs the vLLM path (--vllm).")
    client = VLLMClient(
        model=args.vllm_model,
        gpu_memory_utilization=args.vllm_gpu_memory,
        seed=seed,
    )
    print(f"Verifying records with Qwen via vLLM: {args.vllm_model}")
    return QwenLabelVerifier(
        client,
        checkpoint=checkpoint,
        batch_size=args.verify_batch_size,
    )


def _apply_relabel(
    records: list[dict],
    verifier,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Re-derive labels and answers, then drop records that fail the checks.

    Returns the surviving records and the list of ``(QUERY_ID, reason)``
    pairs for the records that were dropped.
    """
    results = verifier.verify(records)
    kept: list[dict] = []
    dropped: list[tuple[str, str]] = []

    for record, result in zip(records, results):
        updated = apply_verification(
            record,
            answerability=result["answerability"],
            reference_answer=result["reference_answer"],
            claim_evidence_map=result["claim_evidence_map"],
        )
        keep, reason = verify_record(updated)
        if keep:
            kept.append(updated)
        else:
            dropped.append((str(record.get("QUERY_ID", "?")), reason))

    return kept, dropped


def _report_dropped(scope: str, dropped: list[tuple[str, str]]) -> None:
    if not dropped:
        return
    print(f"  -> --relabel: dropped {len(dropped)} {scope} record(s):")
    for query_id, reason in dropped[:20]:
        print(f"       {query_id}: {reason}")
    if len(dropped) > 20:
        print(f"       ... and {len(dropped) - 20} more")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.limit < 0:
        print("[ERROR] --limit must be 0 (all) or a positive integer.", file=sys.stderr)
        return 1

    if not args.chunks.exists():
        print(f"[ERROR] chunks parquet not found at {args.chunks}", file=sys.stderr)
        return 1

    print(f"Loading chunks from {args.chunks} ...")
    chunks = load_chunks(args.chunks)
    print(
        f"  -> {len(chunks)} chunks across "
        f"{len({c.doc_id for c in chunks})} documents"
    )

    vaccine_mapping: dict[str, str] = {}
    if args.mapping.exists():
        with open(args.mapping, "r", encoding="utf-8") as f:
            vaccine_mapping = json.load(f)
        print(f"  -> vaccine name mapping loaded from {args.mapping}")

    seed = (
        args.seed
        if args.seed is not None
        else derive_default_seed(chunks, args.per_type or 3)
    )
    rng = random.Random(seed)
    print(f"Using seed: {seed}")

    # One checkpoint store shared by all three LLM stages (rewrites,
    # answers, symptom masks) so a single file resumes the whole run.
    checkpoint: VLLMCheckpoint | None = None
    if args.checkpoint is not None and _resolve_use_vllm(args):
        checkpoint = VLLMCheckpoint(
            args.checkpoint, model=args.vllm_model, seed=seed
        )
        try:
            checkpoint.load()
        except VLLMError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    if args.source == "queries":
        try:
            queries, rpt = _run_queries_source(
                args, chunks, rng, vaccine_mapping, seed, checkpoint
            )
        except VLLMError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        print(f"  -> {len(queries)} queries generated")
    else:
        per_type = args.per_type if args.per_type > 0 else 3
        if args.generate_answers:
            if args.dry_run_answers:
                ans_gen = ExtractiveAnswerGenerator()
            else:
                ans_gen = OpenRouterAnswerGenerator(model=args.answers_model)
        else:
            ans_gen = NoOpAnswerGenerator()

        print(
            f"Generating {per_type} query/QUERY_TYPE with {args.distractors} "
            f"distractors per query ..."
        )
        queries, rpt = generate_and_validate(
            chunks, rng, per_type, args.distractors,
            vaccine_mapping=vaccine_mapping,
            answer_generator=ans_gen,
            expected_per_type=per_type,
        )
        print(f"  -> {len(queries)} queries generated")

    if not args.no_validate:
        print(rpt.render())
        if not rpt.ok:
            print("[ERROR] validation failed; not writing dataset.", file=sys.stderr)
            return 2
        if args.strict_validation and rpt.warnings:
            print(
                f"[ERROR] --strict-validation: {len(rpt.warnings)} warnings "
                "treated as failures.",
                file=sys.stderr,
            )
            return 2

    # Optional post-processing, applied after validation so the written
    # records remain schema-valid.
    if args.filter_no_han:
        before = len(queries)
        queries = filter_no_han(queries)
        print(f"  -> --filter-no-han: dropped {before - len(queries)} records")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.mask_symptoms:
        head, tail = split_records(queries, args.split_at)
        masker = _build_masker(args, chunks, rng, checkpoint, seed)
        pairs = masker.mask(tail)

        masked_tail: list[dict] = []
        applied_total = 0
        for record, record_pairs in zip(tail, pairs):
            updated, report = apply_replacements(record, record_pairs)
            applied_total += len(report["applied"])
            masked_tail.append(updated)

        # The masker rewrote the queries, so the labels and answers of the
        # tail (and of the head, when it is covered too) describe the old
        # question. Re-derive them from the final queries.
        if args.relabel:
            verifier = _build_verifier(args, chunks, rng, checkpoint, seed)
            head_dropped: list[tuple[str, str]] = []
            if args.relabel_scope == "all":
                head, head_dropped = _apply_relabel(head, verifier)
            masked_tail, tail_dropped = _apply_relabel(masked_tail, verifier)
            _report_dropped("head", head_dropped)
            _report_dropped("masked", tail_dropped)

        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(head, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(head)} unmasked records to {args.out}")

        out_masked = args.out_masked
        if out_masked is None:
            out_masked = args.out.with_name(args.out.stem + "_masked.json")
        out_masked.parent.mkdir(parents=True, exist_ok=True)
        with open(out_masked, "w", encoding="utf-8") as f:
            json.dump(masked_tail, f, ensure_ascii=False, indent=2)
        print(
            f"Wrote {len(masked_tail)} symptom-masked records to {out_masked} "
            f"({applied_total} symptoms replaced)"
        )
        return 0

    if args.relabel:
        verifier = _build_verifier(args, chunks, rng, checkpoint, seed)
        queries, dropped = _apply_relabel(queries, verifier)
        _report_dropped("dataset", dropped)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)
    print(f"Wrote dataset to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
