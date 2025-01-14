import pytest
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

from pages import search_terms
from pages.search_terms import main

def test_search_terms_page_loads(mock_streamlit):
    """Test if search terms page loads without errors"""
    try:
        search_terms.main()
        assert True
    except Exception as e:
        pytest.fail(f"Search terms page failed to load: {str(e)}")

def test_search_terms_interface(mock_streamlit):
    """Test if search terms interface elements are present"""
    search_terms.main()

    # Verify title is set
    mock_streamlit.title.assert_called_once_with("🔑 Search Terms")

    # Verify subheaders are created
    mock_streamlit.subheader.assert_any_call("Existing Search Terms")
    mock_streamlit.subheader.assert_any_call("Add New Search Term")
    mock_streamlit.subheader.assert_any_call("Delete Search Terms")

def test_add_search_term(mock_streamlit, mock_db):
    """Test adding a new search term"""
    # Mock form inputs
    mock_streamlit.text_input.return_value = "new test term"
    mock_streamlit.selectbox.return_value = "ES"

    # Run the main function
    search_terms.main()

    # Verify form elements were created
    mock_streamlit.text_input.assert_called_with("Enter new search term:")
    mock_streamlit.selectbox.assert_called_with("Language", ['ES', 'EN'], index=0)