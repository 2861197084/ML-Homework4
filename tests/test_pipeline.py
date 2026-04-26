from __future__ import annotations

from pathlib import Path

from adult_lab.pipeline import run_all


def test_run_all_pipeline_generates_expected_outputs(mini_project: Path) -> None:
    outputs = run_all(mini_project)

    for key in [
        "metrics_path",
        "predictions_path",
        "tree_model_path",
        "forest_model_path",
        "report_path",
    ]:
        assert Path(outputs[key]).exists()

    assert Path(outputs["figure_paths"]["roc_curves"]).exists()
    assert Path(outputs["figure_paths"]["forest_grid_heatmap"]).exists()
