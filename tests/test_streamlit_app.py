"""End-to-end headless UI integration test for the 4-tab Streamlit application."""

from pathlib import Path
from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app" / "streamlit_app.py"


def test_streamlit_dashboard_renders_all_tabs_and_sql_query() -> None:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()

    assert len(at.exception) == 0, f"Unexpected UI exception: {at.exception}"
    assert len(at.error) == 0, f"Unexpected UI error: {at.error}"
    assert len(at.tabs) == 4
    assert len(at.metric) == 5

    # Click 'Run DuckDB Query' in Tab 4 and verify dataframe renders without error
    at.button[0].click().run()
    assert len(at.exception) == 0
    assert len(at.dataframe) == 2
