import pytest
from debugai.utils.metrics import calculate_metrics
from debugai.utils.code_parser import parse_code

def test_metrics_calculation(sample_py_file):
    with open(sample_py_file, 'r') as f:
        content = f.read()
    
    metrics = calculate_metrics(content)
    assert 'total_lines' in metrics
    assert 'code_lines' in metrics
    assert 'comment_lines' in metrics
    assert metrics['total_lines'] > 0

def test_code_parsing(sample_py_file):
    with open(sample_py_file, 'r') as f:
        content = f.read()
    
    parsed = parse_code(content)
    assert 'ast' in parsed
    assert 'type' in parsed
    assert parsed['type'] == 'Module' 