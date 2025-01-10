from playwright.sync_api import sync_playwright, Page, Browser
from typing import Optional, Dict, Any, Union
import time
import base64
import requests
import json
from pathlib import Path
import os

class BrowserAction:
    def __init__(self, headless: bool = False, hf_token: str = None):
        self.playwright = sync_playwright().start()
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.headless = headless
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        
    def start(self) -> None:
        """Start browser session"""
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()
        
    def goto(self, url: str) -> None:
        """Navigate to URL"""
        if not self.page:
            self.start()
        self.page.goto(url)
        
    def click(self, selector: str) -> None:
        """Click element by selector"""
        self.page.click(selector)
        
    def type(self, selector: str, text: str) -> None:
        """Type text into element"""
        self.page.type(selector, text)
        
    def get_text(self, selector: str) -> str:
        """Get text content of element"""
        return self.page.text_content(selector)
    
    def wait_for_selector(self, selector: str, timeout: int = 30000) -> None:
        """Wait for element to be visible"""
        self.page.wait_for_selector(selector, timeout=timeout)
        
    def screenshot(self, path: str = None, selector: str = None, analyze: bool = False) -> Union[str, Dict]:
        """Take screenshot and optionally analyze it
        
        Args:
            path: Path to save screenshot
            selector: CSS selector to screenshot specific element
            analyze: Whether to analyze screenshot with vision model
            
        Returns:
            Path to screenshot or analysis results if analyze=True
        """
        if selector:
            element = self.page.query_selector(selector)
            if path:
                element.screenshot(path=path)
            else:
                path = f"temp_screenshot_{int(time.time())}.png"
                element.screenshot(path=path)
        else:
            if path:
                self.page.screenshot(path=path)
            else:
                path = f"temp_screenshot_{int(time.time())}.png"
                self.page.screenshot(path=path)
                
        if analyze:
            return self.analyze_image(path)
        return path
            
    def analyze_image(self, image_path: str, prompt: str = None) -> Dict:
        """Analyze image using Hugging Face Vision model
        
        Args:
            image_path: Path to image file
            prompt: Custom prompt for analysis. If None, uses default descriptive prompt
            
        Returns:
            Dict containing analysis results
        """
        if not self.hf_token:
            raise ValueError("Hugging Face token required for image analysis")
            
        # Read and encode image
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
            
        # Default analytical prompt if none provided
        if not prompt:
            prompt = f"""Analyze this image in detail and provide:
1. A comprehensive description of what you see
2. Any notable elements, patterns or features
3. Relevant measurements or coordinates if applicable
4. Any potential issues or anomalies

<image>data:image/png;base64,{img_data}</image>"""

        # Call Hugging Face API
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.2
            }
        }
        
        response = requests.post(
            "https://api-inference.huggingface.co/models/meta-llama/Llama-2-70b-chat-hf",
            headers=headers,
            json=payload
        )
        
        return response.json()
        
    def evaluate(self, js_code: str) -> Any:
        """Execute JavaScript code"""
        return self.page.evaluate(js_code)
        
    def fill_form(self, form_data: Dict[str, str]) -> None:
        """Fill form fields with provided data"""
        for selector, value in form_data.items():
            self.type(selector, value)
            time.sleep(0.5)
            
    def close(self) -> None:
        """Close browser session"""
        if self.browser:
            self.browser.close()
        self.playwright.stop()
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 