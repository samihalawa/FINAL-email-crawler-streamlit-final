import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from pages import health_dashboard
from health_checks import display_health_dashboard

def test_health_dashboard_page_loads(mock_streamlit):
    """Test if health dashboard page loads without errors"""
    try:
        health_dashboard.display_health_dashboard()
        assert True
    except Exception as e:
        pytest.fail(f"Health dashboard page failed to load: {str(e)}")

def test_health_dashboard_interface(mock_streamlit):
    """Test if health dashboard interface elements are present"""
    display_health_dashboard()
    
    # Verify title is set
    mock_streamlit.title.assert_called_once_with("🏥 System Health Dashboard")
    
    # Verify key UI elements
    mock_streamlit.checkbox.assert_called_once_with("Auto-refresh (30s)", value=False, key="health_refresh")
    mock_streamlit.button.assert_called_once_with("Run Health Check", key="run_health_check")

def test_health_checks(mock_streamlit, mock_db):
    """Test health check functions"""
    # Run health checks
    display_health_dashboard()
    
    # Verify subheaders are created
    mock_streamlit.subheader.assert_any_call("System Resources")
    mock_streamlit.subheader.assert_any_call("Database Status")
    mock_streamlit.subheader.assert_any_call("Background Processes")
    mock_streamlit.subheader.assert_any_call("Page Status")
