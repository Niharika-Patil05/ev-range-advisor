import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.harness import (file_sha256, git_revision, run_experiment,
                                    summarise_across_seeds)
from src.evaluation.provenance import ConclusionType, Evidence, Provenance, ProvenanceError


def fake_experiment(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"model": ["a", "b"], "MAPE": rng.uniform(1, 2, 2)})


def evidence(**kw):
    base = dict(experiment="E_test", measured=["HV battery current"],
                derived=["wh_per_km"], split_protocol="L1 unseen routes", n="120 segments")
    base.update(kw)
    return Evidence(**base)


def test_run_experiment_writes_the_full_record(tmp_path):
    res = run_experiment("E_test", "smoke", fake_experiment, evidence(), tmp_path,
                         seeds=(0, 1, 2), verbose=False)
    out = res["out_dir"]
    for name in ("metrics_per_seed.csv", "metrics_summary.csv", "MANIFEST.json", "EVIDENCE.md"):
        assert (out / name).exists(), name


def test_every_seed_is_recorded(tmp_path):
    res = run_experiment("E_test", "smoke", fake_experiment, evidence(), tmp_path,
                         seeds=(0, 1, 2), verbose=False)
    assert sorted(res["per_seed"]["seed"].unique()) == [0, 1, 2]
    assert len(res["per_seed"]) == 6


def test_summary_reports_mean_and_std_not_a_single_run(tmp_path):
    res = run_experiment("E_test", "smoke", fake_experiment, evidence(), tmp_path,
                         seeds=(0, 1, 2), verbose=False)
    cols = res["summary"].columns
    assert {"MAPE_mean", "MAPE_std", "MAPE_count"} <= set(cols)
    assert (res["summary"]["MAPE_count"] == 3).all()


def test_manifest_records_provenance_and_revision(tmp_path):
    res = run_experiment("E_test", "smoke", fake_experiment, evidence(), tmp_path,
                         seeds=(0,), config={"alpha": 0.1}, verbose=False)
    m = json.loads((res["out_dir"] / "MANIFEST.json").read_text())
    assert m["provenance"] == "REAL"
    assert m["config"] == {"alpha": 0.1}
    assert m["git_revision"]
    assert m["split_protocol"] == "L1 unseen routes"


def test_synthetic_run_cannot_be_filed_as_demonstrated(tmp_path):
    with pytest.raises(ProvenanceError):
        run_experiment("E_bad", "should fail", fake_experiment,
                       evidence(provenance=Provenance.SYNTHETIC,
                                conclusion_type=ConclusionType.DEMONSTRATED),
                       tmp_path, seeds=(0,), verbose=False)


def test_data_files_are_hashed_into_the_manifest(tmp_path):
    data = tmp_path / "d.csv"
    data.write_text("a,b\n1,2\n")
    res = run_experiment("E_test", "smoke", fake_experiment, evidence(), tmp_path,
                         seeds=(0,), data_files=[data], verbose=False)
    m = json.loads((res["out_dir"] / "MANIFEST.json").read_text())
    assert m["data_files"][str(data)] == file_sha256(data)


def test_summarise_across_seeds_groups_by_index():
    per_seed = pd.concat([fake_experiment(s).assign(seed=s) for s in (0, 1)], ignore_index=True)
    out = summarise_across_seeds(per_seed, ["model"])
    assert sorted(out["model"]) == ["a", "b"]


def test_git_revision_is_reported():
    assert "UNKNOWN" not in git_revision()
