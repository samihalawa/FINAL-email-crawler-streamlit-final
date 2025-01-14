import pytest
from app import *

def test_main_app_loads(mock_streamlit):
    """Test if main app loads without errors"""
    try:
        # Run the main app
        main()
        assert True  # If we get here without errors, test passes
    except Exception as e:
        pytest.fail(f"Main app failed to load: {str(e)}")

def test_app_title(mock_streamlit):
    """Test if app title is set"""
    main()
    mock_streamlit.title.assert_called_once_with("Welcome to AutoclientAI")