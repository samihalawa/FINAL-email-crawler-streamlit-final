from debugai.core.mapper import CodeMapper
import pytest

def test_mapper_class_detection(sample_py_file):
    mapper = CodeMapper()
    analysis = mapper.analyze_file(sample_py_file)
    
    assert len(analysis['classes']) == 1
    assert analysis['classes'][0]['name'] == 'DataAnalyzer'
    assert 'analyze' in [m['name'] for m in analysis['classes'][0]['methods']]

def test_mapper_function_detection(sample_py_file):
    mapper = CodeMapper()
    analysis = mapper.analyze_file(sample_py_file)
    
    assert len(analysis['functions']) == 1
    assert analysis['functions'][0]['name'] == 'process_data'

def test_mapper_component_detection(sample_py_file):
    mapper = CodeMapper(show_components=True)
    analysis = mapper.analyze_file(sample_py_file)
    
    assert 'components' in analysis
    assert 'streamlit' in analysis['components']
    assert any('st.title' in comp['type'] for comp in analysis['components']['streamlit']) 