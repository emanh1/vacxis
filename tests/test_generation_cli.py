import json
from unittest.mock import Mock

import pytest

from vacxis_capstone_fa26.generation.cli import main


def test_cli_ignores_planner_env_for_online_queries(monkeypatch, capsys):
    from vacxis_capstone_fa26.generation import cli
    from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer
    monkeypatch.setenv("VACXIS_QUERY_PLANNER_MODEL", "planner-from-env")
    monkeypatch.setenv("VACXIS_GENERATION_MODEL", "generation-from-env")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.setattr("sys.argv", ["vacxis-generation", "--query", "Verorab là gì?", "--json"])
    from contextlib import contextmanager
    @contextmanager
    def fake_retriever(args):
        yield Mock()
    monkeypatch.setattr(cli, "build_retriever", fake_retriever)
    pipeline = Mock()
    pipeline.answer.return_value = GeneratedAnswer(query="Verorab là gì?", answer="Answer")
    factory = Mock(return_value=pipeline)
    monkeypatch.setattr(cli, "VacxisRAGPipeline", factory)
    main()
    assert "query_planner" not in factory.call_args.kwargs
    assert factory.call_args.kwargs["generator"].config.model == "generation-from-env"
    pipeline.run_turn.assert_not_called()
    assert json.loads(capsys.readouterr().out)["answer"] == "Answer"


@pytest.mark.parametrize("arguments", [
    ["--query-planner-model", "fixture", "--dry-run"],
    ["--query-planner-model", "fixture", "--input-file", "test_dataset.json"],
    ["--clarification", "question", "answer"],
])
def test_cli_rejects_incompatible_modes(monkeypatch, arguments):
    monkeypatch.delenv("VACXIS_QUERY_PLANNER_MODEL", raising=False)
    monkeypatch.setattr("sys.argv", ["vacxis-generation", *arguments])
    with pytest.raises(SystemExit, match="2"):
        main()


@pytest.mark.parametrize("arguments", [
    ["--dry-run", "--query", "Verorab là gì?"],
    ["--input-file", "test_dataset.json"],
])
def test_cli_ignores_planner_env_outside_single_online_mode(monkeypatch, arguments):
    from vacxis_capstone_fa26.generation.cli import parse_args

    monkeypatch.setenv("VACXIS_QUERY_PLANNER_MODEL", "planner-from-env")
    monkeypatch.setattr("sys.argv", ["vacxis-generation", *arguments])

    assert not hasattr(parse_args(), "query_planner_model")


@pytest.mark.parametrize('variant', [[], ['--baseline'], ['--advanced-rag']])
def test_batch_checkpoints_and_resumes_all_variants(monkeypatch, tmp_path, variant):
    from contextlib import contextmanager
    from vacxis_capstone_fa26.generation import cli
    from vacxis_capstone_fa26.generation.schemas import GeneratedAnswer
    dataset = tmp_path / 'dataset.json'
    dataset.write_text(json.dumps({'goldens': [
        {'input': 'q1', 'additional_metadata': {'query_id': '1'}},
        {'input': 'q2', 'additional_metadata': {'query_id': '2'}},
    ]}))
    output = tmp_path / 'answers.json'
    @contextmanager
    def fake_retriever(args):
        yield Mock()
    monkeypatch.setattr(cli, 'build_retriever', fake_retriever)
    pipeline = Mock()
    pipeline.answer.side_effect = [GeneratedAnswer(query='q1', answer='a1'), RuntimeError('dropped')]
    monkeypatch.setattr(cli, 'VacxisRAGPipeline', Mock(return_value=pipeline))
    argv = ['vacxis-generation', '--dry-run', '--input-file', str(dataset), '--out-file', str(output), *variant]
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(RuntimeError, match='dropped'):
        cli.main()
    assert len(json.loads(output.read_text())) == 1
    pipeline.answer.reset_mock(side_effect=True)
    pipeline.answer.return_value = GeneratedAnswer(query='q2', answer='a2')
    monkeypatch.setattr('sys.argv', argv + ['--resume'])
    cli.main()
    assert pipeline.answer.call_count == 1
    assert len(json.loads(output.read_text())) == 2
    monkeypatch.setattr('sys.argv', argv + ['--resume', '--top-k', '9'])
    with pytest.raises(ValueError, match='settings changed'):
        cli.main()


def test_fresh_failed_run_cannot_resume_old_answers(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from vacxis_capstone_fa26.generation import cli
    dataset = tmp_path / 'dataset.json'
    dataset.write_text(json.dumps({'goldens': [{'input': 'new', 'additional_metadata': {'query_id': '1'}}]}))
    output = tmp_path / 'answers.json'
    output.write_text(json.dumps([{'additional_metadata': {'query_id': '1'}, 'GENERATED_ANSWER': {'answer': 'old'}}]))
    @contextmanager
    def fail(args):
        raise RuntimeError('offline')
        yield
    monkeypatch.setattr(cli, 'build_retriever', fail)
    monkeypatch.setattr('sys.argv', ['vacxis-generation', '--dry-run', '--baseline', '--input-file', str(dataset), '--out-file', str(output)])
    with pytest.raises(RuntimeError):
        cli.main()
    assert json.loads(output.read_text()) == []


def test_call_delay_does_not_invalidate_existing_manifest(monkeypatch, tmp_path):
    from vacxis_capstone_fa26.generation import cli
    dataset = tmp_path / 'dataset.json'
    dataset.write_text(json.dumps({'goldens': [
        {'input': 'question', 'additional_metadata': {'query_id': '1'}}
    ]}))
    output = tmp_path / 'answers.json'
    argv = ['vacxis-generation', '--dry-run', '--use-ground-truth-chunks',
            '--input-file', str(dataset), '--out-file', str(output)]
    monkeypatch.setattr('sys.argv', argv)
    cli.main()
    # A pre-pacing manifest has the same signature: the new flag is excluded.
    monkeypatch.setattr('sys.argv', argv + ['--resume', '--llm-call-delay', '10'])
    cli.main()
    assert len(json.loads(output.read_text())) == 1
