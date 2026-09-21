"""Smoke test for the Streamlit app.

Cheap insurance: the app imports most of the codebase, so a signature change
anywhere tends to break it, and nothing else in the suite would notice.
"""
from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

# Resolve from this file, not the working directory: pytest may be invoked from
# anywhere, and a relative path silently becomes FileNotFoundError.
APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    return at


def test_app_runs_without_exception(app):
    assert not app.exception, app.exception[0].value if app.exception else ""


def test_app_has_the_four_tabs(app):
    assert [t.label for t in app.tabs] == [
        "Trip advisor", "Battery health", "Evidence", "Sensitivity"]


def test_app_states_the_provenance_of_the_model_it_is_serving(app):
    """Whichever model is loaded, the app must say which. The headline results in
    reports/ are real; a viewer must never be left to assume the demo produced them.

    Real artefact  -> a success banner naming the data and the measured accuracy.
    Synthetic only -> a warning that these are NOT the reported results.
    """
    banners = " ".join(m.value for m in list(app.success) + list(app.warning))
    assert ("REAL logged vehicle data" in banners) or ("SYNTHETIC model" in banners)


def test_real_banner_quotes_the_measured_holdout_error(app):
    """If the real model is being served, the interface must show what it actually
    scores on unseen routes rather than implying the headline figure."""
    banners = " ".join(m.value for m in app.success)
    if "REAL logged vehicle data" not in banners:
        pytest.skip("synthetic fallback in use; nothing to check")
    assert "MAPE" in banners and "unseen routes" in banners
