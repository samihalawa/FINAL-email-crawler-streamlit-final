from click.testing import CliRunner
from debugai.cli.commands import cli
import pytest

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'Usage:' in result.output

def test_analyze_command(sample_py_file):
    runner = CliRunner()
    result = runner.invoke(cli, ['analyze', sample_py_file])
    assert result.exit_code == 0
    assert 'DataAnalyzer' in result.output
    assert 'process_data' in result.output 