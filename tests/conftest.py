import pytest
from pathlib import Path
import streamlit as st
import sys

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def mock_streamlit(mocker):
    """Mock common Streamlit commands"""
    # Create MagicMock objects for each Streamlit function
    title_mock = mocker.patch('streamlit.title')
    header_mock = mocker.patch('streamlit.header')
    subheader_mock = mocker.patch('streamlit.subheader')
    sidebar_mock = mocker.patch('streamlit.sidebar')
    text_mock = mocker.patch('streamlit.text')
    write_mock = mocker.patch('streamlit.write')
    text_input_mock = mocker.patch('streamlit.text_input')
    button_mock = mocker.patch('streamlit.button')
    slider_mock = mocker.patch('streamlit.slider')
    selectbox_mock = mocker.patch('streamlit.selectbox')
    form_mock = mocker.patch('streamlit.form')
    spinner_mock = mocker.patch('streamlit.spinner')
    info_mock = mocker.patch('streamlit.info')
    success_mock = mocker.patch('streamlit.success')
    error_mock = mocker.patch('streamlit.error')
    warning_mock = mocker.patch('streamlit.warning')
    dataframe_mock = mocker.patch('streamlit.dataframe')

    # Return st module with mocked functions
    return st

@pytest.fixture
def mock_db(mocker):
    """Mock database connection"""
    mock_engine = mocker.Mock()
    mock_session = mocker.Mock()
    mocker.patch('sqlalchemy.create_engine', return_value=mock_engine)
    mocker.patch('sqlalchemy.orm.sessionmaker', return_value=lambda: mock_session)
    return mock_engine