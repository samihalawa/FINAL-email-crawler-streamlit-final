import pytest
from debugai.core.mapper import CodeMapper
from debugai.utils.formatters import format_output

def test_full_analysis_flow(sample_py_file):
    # Initialize mapper
    mapper = CodeMapper(show_line_numbers=True, show_components=True)
    
    # Analyze file
    results = {sample_py_file: mapper.analyze_file(sample_py_file)}
    
    # Test different output formats
    text_output = format_output(results, 'text')
    json_output = format_output(results, 'json')
    md_output = format_output(results, 'md')
    
    # Verify outputs
    assert 'TestClass' in text_output
    assert 'method1' in text_output
    assert 'method2' in text_output
    assert 'main' in text_output 