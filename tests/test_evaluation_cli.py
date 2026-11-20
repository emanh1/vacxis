"""Shared evaluation CLI contracts without live retrieval."""
from unittest.mock import Mock


def test_full_evaluation_records_unanswerable_dataset(monkeypatch, tmp_path):
    from vacxis_capstone_fa26.evaluation import cli
    from vacxis_capstone_fa26.evaluation.schemas import EvaluationSummary

    monkeypatch.setattr("sys.argv", ["vacxis-eval", "--unanswerable", str(tmp_path / "unanswerable.json")])
    args = cli.parse_args()
    summaries = [EvaluationSummary(metadata={"run": {}}), EvaluationSummary(metadata={"run": {}})]
    monkeypatch.setattr(cli, "_evaluate_live", Mock(side_effect=summaries))
    monkeypatch.setattr(cli, "_load_dataset_items", Mock(return_value=[{}]))
    monkeypatch.setattr(cli, "_load_flat_dataset", Mock(return_value=[{}]))
    monkeypatch.setattr(cli, "_print_summary", Mock())
    monkeypatch.setattr(cli, "_maybe_export", Mock())
    cli._run_live(Mock(), args, {})
    assert summaries[1].metadata["run"]["dataset"] == str(args.unanswerable)


def test_skip_unanswerable_does_not_read_auxiliary_file(monkeypatch, tmp_path):
    from vacxis_capstone_fa26.evaluation import cli
    from vacxis_capstone_fa26.evaluation.schemas import EvaluationSummary
    results = tmp_path / 'results.json'
    results.write_text('[]')
    monkeypatch.setattr('sys.argv', ['vacxis-eval', '--retrieval-only', '--results', str(results),
        '--skip-unanswerable', '--unanswerable', str(tmp_path / 'missing.json')])
    evaluator = Mock()
    evaluator.evaluate_retrieval.return_value = EvaluationSummary()
    monkeypatch.setattr(cli, 'VacxisRAGEvaluator', Mock(return_value=evaluator))
    monkeypatch.setattr(cli, '_print_summary', Mock())
    monkeypatch.setattr(cli, '_maybe_export', Mock())
    cli.main()
    evaluator.evaluate_retrieval.assert_called_once()
