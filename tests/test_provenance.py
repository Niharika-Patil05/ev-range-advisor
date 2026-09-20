import pandas as pd
import pytest

from src.evaluation.provenance import (ConclusionType, Evidence, Provenance,
                                       ProvenanceError, provenances_of,
                                       require_single_provenance, tag)


def _df(n=3):
    return pd.DataFrame({"wh_per_km": [150.0] * n})


def test_tag_adds_provenance_and_source():
    out = tag(_df(), Provenance.REAL, "VED")
    assert set(out["provenance"]) == {"REAL"}
    assert set(out["source_dataset"]) == {"VED"}


def test_untagged_frame_is_rejected():
    with pytest.raises(ProvenanceError, match="no 'provenance' column"):
        provenances_of(_df())


def test_mixing_real_and_synthetic_is_refused():
    mixed = pd.concat([tag(_df(), Provenance.REAL, "VED"),
                       tag(_df(), Provenance.SYNTHETIC, "generator")])
    with pytest.raises(ProvenanceError, match="different provenance"):
        require_single_provenance(mixed)


def test_mixing_allowed_only_when_explicit():
    mixed = pd.concat([tag(_df(), Provenance.REAL, "VED"),
                       tag(_df(), Provenance.SYNTHETIC, "generator")])
    # the weaker provenance wins, so a mixed table can never be called REAL
    assert require_single_provenance(mixed, allow_mixed=True) is Provenance.SYNTHETIC


def test_single_provenance_passes_through():
    assert require_single_provenance(tag(_df(), Provenance.REAL, "VED")) is Provenance.REAL


def test_synthetic_data_cannot_support_a_demonstrated_claim():
    with pytest.raises(ProvenanceError, match="Cannot report a DEMONSTRATED"):
        Evidence(experiment="E1", provenance=Provenance.SYNTHETIC,
                 conclusion_type=ConclusionType.DEMONSTRATED)


def test_coupled_sim_cannot_support_a_demonstrated_claim():
    with pytest.raises(ProvenanceError):
        Evidence(experiment="E6", provenance=Provenance.COUPLED_SIM,
                 conclusion_type=ConclusionType.DEMONSTRATED)


def test_coupled_sim_markdown_carries_the_warning():
    md = Evidence(experiment="E6", provenance=Provenance.COUPLED_SIM,
                  conclusion_type=ConclusionType.SIMULATED,
                  measured=["NASA cell capacity"],
                  assumptions=["temperature derate form"]).to_markdown()
    assert "COUPLED-SIM" in md
    assert "not** real-world end-to-end validation" in md


def test_association_only_note_is_emitted():
    md = Evidence(experiment="E3", provenance=Provenance.REAL,
                  conclusion_type=ConclusionType.DEMONSTRATED).to_markdown()
    assert "No causal claim is made" in md
