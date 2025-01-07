import logging
import pytest
from playwright.sync_api import sync_playwright
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_ui_components():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        try:
            page.goto("http://localhost:8501")
            page.wait_for_load_state("networkidle")
            time.sleep(5)  # Wait for Streamlit to fully load
            page.screenshot(path="debug_initial_load.png")
            page.wait_for_selector("[data-testid=stSidebar]", timeout=10000)
            page.click("text=Manual Search")
            page.wait_for_selector("input[type=text]")
            page.fill("input[type=text]", "test")
            page.click("text=Search")
            logger.info("Tests completed")
        except Exception as e:
            logger.error(str(e))
            page.screenshot(path="debug_error.png")
            raise
        finally:
            browser.close()

if __name__ == "__main__":
    test_ui_components()