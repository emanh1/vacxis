"""Offline cross-module integration and orchestration regression tests."""
from dataclasses import replace
import json
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from qdrant_client import QdrantClient

from vacxis_capstone_fa26 import cli, workflow
from vacxis_capstone_fa26.evaluation.evaluator import VacxisRAGEvaluator
from vacxis_capstone_fa26.generation.generator import MedicalGenerator
from vacxis_capstone_fa26.generation.pipeline import VacxisRAGPipeline
from vacxis_capstone_fa26.generation.schemas import GenerationConfig
from vacxis_capstone_fa26.indexing.indexer import DENSE_DIM, ingest_to_qdrant
from vacxis_capstone_fa26.retrieval.hybrid_searcher import HybridSearcher
from vacxis_capstone_fa26.retrieval.retriever import VacxisRetriever


class FixtureEmbedder:
    """Small deterministic model substitute; the actual Qdrant search still executes."""

    def encode(self, texts, **kwargs):
        return {"dense_vecs": np.tile([1.0] + [0.0] * (DENSE_DIM - 1), (len(texts), 1)),
                "lexical_weights": [{"1": 1.0} for _ in texts]}


@pytest.fixture
def config(tmp_path):
    from vacxis_capstone_fa26.testdata.taxonomy import QUERY_TYPE_DEFS
    categories = list(dict.fromkeys(item["category"] for item in QUERY_TYPE_DEFS))
    for folder_id in ("001", "002"):
        folder = tmp_path / "CLEANED/HUNG" / folder_id
        folder.mkdir(parents=True)
        sections = []
        for category in categories:
            # Long sections stay independent after real chunking, preserving taxonomy coverage.
            content = (f"Thông tin {category}: vắc xin dùng cho người từ 9 tuổi trở lên. "
                       "Tiêm bắp 0,5 ml mỗi liều, cách nhau 6 tháng. Theo dõi phản ứng sau tiêm. ")
            sections.append(f"# {category.upper()}\n" + content * 15)
        (folder / "vaccine.md").write_text("\n\n".join(sections), encoding="utf-8")
    (tmp_path / "75_vaccine_names_mapping.json").write_text(
        '{"001":"Vax A","002":"Vax B"}', encoding="utf-8",
    )
    return workflow.WorkflowConfig(project_root=tmp_path,
                                   generation=GenerationConfig(dry_run=True))


@pytest.mark.parametrize("online", [False, True])
def test_full_offline_workflow(config, monkeypatch, online):
    client = QdrantClient(":memory:")
    embedder = FixtureEmbedder()
    reranker = Mock()
    reranker.batch_rerank.side_effect = lambda queries, candidates_list, top_k, **kw: [
        [replace(c, score=0.9, rerank_score=0.9) for c in group[:top_k]] for group in candidates_list
    ]
    try:
        count = workflow.prepare(config, client=client, embed_model=embedder)
        rows = pq.read_table(config.chunks_path).to_pylist()
        assert count == len(rows) > 0
        chunk_ids = {row["chunk_id"] for row in rows}
        # The dataset stage now lives in the testdata CLI (vLLM/GPU host);
        # the workflow assumes it exists at config.dataset_path.
        items = [
            {
                "QUERY_ID": f"VAX_Q_{i:04d}",
                "QUERY": f"Câu hỏi {i}",
                "REFERENCE_ANSWER": "Đáp án mẫu.",
                "RELEVANT_CHUNKS": [{
                    "CHUNK_ID": row["chunk_id"], "TEXT": row["text"],
                    "RELEVANCE_SCORE": 3,
                }],
            }
            for i, row in enumerate(rows[:2], start=1)
        ]
        workflow.write_json(config.dataset_path, items)
        assert all(c["CHUNK_ID"] in chunk_ids for item in items for c in item["RELEVANT_CHUNKS"])
        searcher = HybridSearcher(client=client, collection_name=config.retrieval.collection_name)
        retriever = VacxisRetriever(config.retrieval, embed_model=embedder, reranker=reranker,
                                   hybrid_searcher=searcher)
        monkeypatch.setattr(workflow, "build_pipeline", lambda settings: VacxisRAGPipeline(
            retriever=retriever, generator=MedicalGenerator(settings.generation),
        ))
        sdk = MagicMock()
        if online:
            from vacxis_capstone_fa26 import llm
            monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
            config.generation = GenerationConfig(model="fixture")
            def respond(**kwargs):
                cid = re.search(r"CHUNK_ID: ([^ |\]]+)", kwargs["messages"][1]["content"]).group(1)
                return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop",
                    message=SimpleNamespace(content=f"Evidence [{cid}]."))], usage=None)
            sdk.return_value.__enter__.return_value.chat.completions.create.side_effect = respond
            monkeypatch.setattr(llm, "OpenAI", sdk)
        summary = workflow.benchmark(config)
        if online:
            assert sdk.call_count == len(items)
            assert sdk.return_value.__exit__.call_count == len(items)
        results = json.loads(config.results_path.read_text())
        assert [r["QUERY_ID"] for r in results] == [q["QUERY_ID"] for q in items]
        assert summary.num_queries == len(items)
        assert summary.task == "rag"
        assert config.report_json.is_file() and config.report_md.is_file()
        assert reranker.batch_rerank.called
        for result in results:
            assert set(result["GENERATED_ANSWER"]["retrieved_chunk_ids"]) <= chunk_ids
            assert result["GENERATED_ANSWER"]["invalid_citations"] == []
        # Reingesting the same corpus retains stable point IDs and collection data.
        ingest_to_qdrant(config.chunks_path, client=client, embed_model=embedder)
        assert client.count(config.retrieval.collection_name).count == count
    finally:
        client.close()


@pytest.mark.parametrize("stage", ["prepare", "benchmark"])
def test_run_stops_at_first_failed_stage(config, monkeypatch, stage):
    calls = []
    for name in ("prepare", "benchmark"):
        def execute(settings, name=name):
            calls.append(name)
            if name == stage:
                raise ValueError("fixture")
        monkeypatch.setattr(workflow, name, execute)
    with pytest.raises(ValueError):
        workflow.run(config)
    assert calls == ["prepare", "benchmark"][:calls.index(stage) + 1]


@pytest.mark.parametrize("command", ["prepare", "ask", "benchmark", "run"])
def test_cli_dispatch_and_shared_options(config, monkeypatch, capsys, command):
    result = {"answer": "fixture"} if command == "ask" else (
        3 if command == "prepare" else SimpleNamespace(num_queries=1)
    )
    action = Mock(return_value=result)
    monkeypatch.setattr(workflow, command, action)
    flags = [command, "--project-root", str(config.project_root), "--collection-name", "custom",
             "--qdrant-host", "server", "--qdrant-port", "7000", "--model-path", "embed",
             "--reranker-path", "rank", "--top-k", "2"]
    if command == "ask":
        flags += ["--query", "Question"]
    assert cli.main(flags) == 0
    settings = action.call_args.args[0]
    assert settings.retrieval.collection_name == "custom"
    assert settings.retrieval.qdrant_host == "server"
    assert settings.retrieval.qdrant_port == 7000
    assert settings.retrieval.embed_model_path == "embed"
    assert settings.retrieval.reranker_model_path == "rank"
    assert settings.recreate is False
    assert capsys.readouterr().out


def test_cli_safe_error_and_recreate_opt_in(config, monkeypatch, capsys):
    action = Mock(side_effect=ValueError("private provider payload"))
    monkeypatch.setattr(workflow, "prepare", action)
    assert cli.main(["prepare", "--project-root", str(config.project_root), "--recreate"]) == 1
    assert action.call_args.args[0].recreate is True
    assert "private" not in capsys.readouterr().err


def test_pipeline_closes_on_failure():
    retriever = Mock(__enter__=Mock(), __exit__=Mock())
    retriever.retrieve.side_effect = ValueError("fixture")
    with pytest.raises(ValueError):
        with VacxisRAGPipeline(retriever=retriever, generator=Mock()) as pipeline:
            pipeline.answer("Question")
    retriever.__exit__.assert_called_once()


def test_indexer_closes_owned_client_on_failure(config, monkeypatch):
    from vacxis_capstone_fa26.indexing import indexer
    from vacxis_capstone_fa26.indexing.main import run_indexing_pipeline
    run_indexing_pipeline(config.project_root, config.chunks_path)
    client = Mock()
    client.get_collections.side_effect = ValueError("fixture")
    monkeypatch.setattr(indexer, "QdrantClient", Mock(return_value=client))
    with pytest.raises(ValueError):
        indexer.ingest_to_qdrant(config.chunks_path, embed_model=FixtureEmbedder())
    client.close.assert_called_once()


def test_generation_and_pipeline_reject_batch_count_mismatch():
    generator = MedicalGenerator(GenerationConfig(dry_run=True))
    with pytest.raises(ValueError):
        generator.generate_batch(["Question"], [])
    with pytest.raises(ValueError):
        VacxisRAGPipeline(retriever=Mock(), generator=generator).answer_batch(
            ["Question"], override_chunks_per_query=[],
        )


@pytest.mark.parametrize("dataset,results", [
    ([{"QUERY_ID": "a"}, {"QUERY_ID": "a"}], [{"QUERY_ID": "a"}]),
    ([{"QUERY_ID": "a"}], [{"QUERY_ID": "a"}, {"QUERY_ID": "a"}]),
    ([{"QUERY_ID": "a"}], [{"QUERY_ID": "b"}]),
    ([{}], [{}]),
    ([{"QUERY_ID": "a"}], []),
])
def test_evaluation_rejects_ambiguous_ids(dataset, results):
    with pytest.raises(ValueError):
        VacxisRAGEvaluator().evaluate_results(dataset, results)


def test_evaluation_preserves_empty_retrieval_and_matches_reordered_ids():
    dataset = [
        {"QUERY_ID": key, "QUERY": key, "REFERENCE_ANSWER": key,
         "RELEVANT_CHUNKS": [{"CHUNK_ID": "c1", "RELEVANCE_SCORE": 3}]}
        for key in ("a", "b")
    ]
    results = [{"QUERY_ID": key, "GENERATED_ANSWER": {"answer": key, "retrieved_chunk_ids": []}}
               for key in ("b", "a")]
    summary = VacxisRAGEvaluator().evaluate_results(dataset, results)
    assert summary.task == "rag"
    assert summary.num_queries == 2
    assert all(row["retrieved_chunk_ids"] == [] for row in summary.per_query_results)
    assert all(row["metrics"]["exact_match"] == 1 for row in summary.per_query_results)
    assert all(row["retrieval_metrics"]["recall@1"] == 0 for row in summary.per_query_results)


def test_benchmark_invalid_input_writes_no_results(config, monkeypatch):
    workflow.write_json(config.dataset_path, [{"QUERY_ID": "a", "QUERY": "Question"}])
    factory = Mock()
    monkeypatch.setattr(workflow, "build_pipeline", factory)
    with pytest.raises(ValueError, match="REFERENCE_ANSWER"):
        workflow.benchmark(config)
    factory.assert_not_called()
    assert not config.results_path.exists()


def test_config_rejects_artifact_collisions(config):
    with pytest.raises(ValueError, match="distinct"):
        workflow.WorkflowConfig(project_root=config.project_root, results_path=config.dataset_path)
