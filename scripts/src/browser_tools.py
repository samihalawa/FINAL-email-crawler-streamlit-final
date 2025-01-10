from playwright.sync_api import sync_playwright
import base64
import requests
import time
import os
from typing import Dict, Any, Optional

def browser_action(
    action: str,
    url: Optional[str] = None,
    selector: Optional[str] = None,
    text: Optional[str] = None,
    screenshot_path: Optional[str] = None,
    analyze: bool = False,
    headless: bool = False,
    wait_time: int = 1000,
    custom_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute browser actions and optionally analyze screenshots.
    
    Args:
        action: The action to perform (goto, click, type, screenshot, analyze)
        url: Target URL for navigation
        selector: CSS selector for element interaction
        text: Text to type or verify
        screenshot_path: Path to save screenshot
        analyze: Whether to analyze screenshot with vision model
        headless: Run browser headlessly
        wait_time: Time to wait for elements in ms
        custom_prompt: Custom prompt for image analysis
        
    Returns:
        Dict containing action results and any analysis
    """
    with sync_playwright() as p:
        # Browser setup
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        result = {"success": True, "data": None, "error": None}
        
        try:
            # Execute requested action
            if action == "goto":
                page.goto(url)
                page.wait_for_load_state("networkidle")
                
            elif action == "click":
                page.wait_for_selector(selector, timeout=wait_time)
                page.click(selector)
                
            elif action == "type":
                page.wait_for_selector(selector, timeout=wait_time)
                page.type(selector, text)
                
            elif action == "screenshot":
                if selector:
                    element = page.wait_for_selector(selector, timeout=wait_time)
                    element.screenshot(path=screenshot_path)
                else:
                    page.screenshot(path=screenshot_path, full_page=True)
                    
                if analyze:
                    # Vision analysis
                    hf_token = os.getenv("HUGGINGFACE_TOKEN")
                    if not hf_token:
                        raise ValueError("HUGGINGFACE_TOKEN environment variable required")
                        
                    with open(screenshot_path, "rb") as img:
                        img_b64 = base64.b64encode(img.read()).decode()
                    
                    # Format prompt exactly like the working curl command
                    vision_prompt = custom_prompt or """Analyze this screenshot and tell me:
1. What elements and text are visible
2. Their positions and styling
3. Any clickable elements or forms"""
                    
                    full_prompt = f"{vision_prompt} <image>data:image/png;base64,{img_b64}</image>"

                    response = requests.post(
                        "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct",
                        headers={
                            "Authorization": f"Bearer {hf_token}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "inputs": full_prompt,
                            "parameters": {
                                "max_new_tokens": 1000,
                                "temperature": 0.1
                            }
                        }
                    )
                    result["data"] = response.json()
            
            # Get page content if no specific data
            if not result["data"]:
                result["data"] = page.content()
                
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            
        finally:
            browser.close()
            
        return result 