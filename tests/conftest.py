import pytest
import tempfile
import os

@pytest.fixture
def sample_py_file():
    content = '''
class TestClass:
    def method1(self):
        pass
        
    def method2(self):
        return True

def main():
    return TestClass()
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture(autouse=True)
def cleanup(sample_py_file):
    yield
    if os.path.exists(sample_py_file):
        os.unlink(sample_py_file) 