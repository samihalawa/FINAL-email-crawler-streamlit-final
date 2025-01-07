import os
import sys
import ast
import time
import base64
import autogen
import logging
from typing_extensions import Annotated
from PIL import ImageGrab

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

config_list = [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY")}]
llm_config = {"temperature": 0.1, "config_list": config_list, "max_tokens": 4000}

try:
    engineer = autogen.AssistantAgent(name="engineer", llm_config=llm_config, 
        system_message="Expert Python engineer focused on debugging and optimizing Streamlit apps")
    user_proxy = autogen.UserProxyAgent(name="user", human_input_mode="NEVER",
        code_execution_config={"use_docker": False, "work_dir": "workspace"})
except Exception as e:
    logging.error(f"Failed to create agents: {str(e)}"); sys.exit(1)

@user_proxy.register_for_execution()
@engineer.register_for_llm(description="Read file contents")
def read_file(file: Annotated[str, "File to read"]) -> tuple:
    try:
        with open(file, 'r') as f:
            return 0, f.read()
    except Exception as e:
        return 1, str(e)

@user_proxy.register_for_execution()
@engineer.register_for_llm(description="Edit and validate code")
def fix_code(file: str, start: int, end: int, code: str) -> tuple:
    try:
        with open(file, 'r') as f:
            lines = f.readlines()
        
        if not (0 < start <= len(lines) and 0 < end <= len(lines)):
            return 1, "Invalid line range"
            
        backup = f"{file}.bak"
        with open(backup, 'w') as f:
            f.writelines(lines)
            
        lines[start-1:end] = [code + "\n"]
        
        try:
            ast.parse("".join(lines))
        except SyntaxError as e:
            with open(file, 'w') as f:
                f.write(open(backup).read())
            return 1, f"Syntax error: {e}"
            
        with open(file, 'w') as f:
            f.writelines(lines)
            
        return 0, "Fix applied successfully"
    except Exception as e:
        return 1, str(e)

def main():
    if len(sys.argv) != 3:
        logging.error("Usage: script.py <file> <task>")
        sys.exit(1)
        
    file, task = sys.argv[1], sys.argv[2]
    
    if not os.path.exists(file):
        logging.error(f"File not found: {file}")
        sys.exit(1)
        
    try:
        user_proxy.initiate_chat(engineer, message=f"""Debug and optimize {file}
Task: {task}

Steps:
1. Analyze code for bugs and performance issues
2. Fix critical bugs
3. Optimize performance bottlenecks
4. Test changes
5. Verify fixes

Begin debugging.""")
    except Exception as e:
        logging.error(f"Debug session failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()