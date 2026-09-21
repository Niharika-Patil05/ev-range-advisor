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


def test_app_warns_that_the_served_model_is_synthetic(app):
    """The demo model is still trained on synthetic data, and the app must say so
    rather than letting a viewer assume the real results are what it is serving."""
    text = " ".join(w.value for w in app.warning)
    assert "SYNTHETIC" in text.upper()
