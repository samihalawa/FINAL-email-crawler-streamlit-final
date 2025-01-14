import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pages import manual_search
from pages.manual_search import perform_search

def test_manual_search_page_loads(mock_streamlit):
    """Test if manual search page loads without errors"""
    try:
        # Run the page script
        manual_search.main()
        assert True
    except Exception as e:
        pytest.fail(f"Manual search page failed to load: {str(e)}")

def test_search_interface_elements(mock_streamlit):
    """Test if search interface elements are present"""
    # Run the main function to create UI elements
    manual_search.main()

    # Verify that title is set
    mock_streamlit.title.assert_called_once_with("🔍 Manual Search")

    # Verify slider is created with correct parameters
    mock_streamlit.slider.assert_called_once_with("Number of results per term", 5, 50, 10)

    # Verify button is created
    mock_streamlit.button.assert_called_once_with("Search")

def test_perform_search(mock_streamlit, mock_db):
    """Test the perform_search function"""
    terms = ["test term"]
    num_results = 10

    # Call perform_search
    results = perform_search(terms, num_results)

    # Assert results structure
    assert isinstance(results, dict)
    assert 'results' in results
    assert 'total_leads' in results