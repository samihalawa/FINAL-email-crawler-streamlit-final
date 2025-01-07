from playwright.sync_api import sync_playwright, expect
import time
import logging
from urllib.request import urlopen
from urllib.error import URLError
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def wait_for_streamlit(url="http://localhost:8501", timeout=60):
    """Wait for Streamlit to be ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = urlopen(url)
            content = response.read().decode('utf-8')
            if 'streamlit-app' in content.lower():
                logger.info("Streamlit is ready!")
                return True
            logger.info("Waiting for Streamlit app to load...")
        except URLError as e:
            logger.info(f"Waiting for Streamlit to be ready... ({str(e)})")
        except Exception as e:
            logger.warning(f"Unexpected error while waiting: {str(e)}")
        time.sleep(2)
    return False

def wait_for_element(page, selector, timeout=30000, state="visible"):
    """Wait for an element to be visible with retries"""
    try:
        page.wait_for_selector(selector, timeout=timeout, state=state)
        return True
    except Exception as e:
        logger.warning(f"Error waiting for {selector}: {str(e)}")
        return False

def retry_on_error(func):
    """Decorator to retry functions on error"""
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                time.sleep(2)
    return wrapper

@retry_on_error
def test_manual_search(page):
    """Test the Manual Search page functionality"""
    logger.info("Testing Manual Search page...")
    
    # Wait for and click the Manual Search tab if not already there
    page.click('text="Manual Search"')
    page.wait_for_selector("h1:has-text('Manual Search')", timeout=10000)
    
    # Input search terms
    page.click("text=Enter search terms:")
    page.keyboard.type("software engineer")
    page.keyboard.press("Enter")
    logger.info("Entered search term")
    
    # Configure search options
    page.click("text=Enable email sending")
    page.click("text=Ignore fetched domains")
    
    # Open advanced options
    page.click("text=Advanced Options")
    time.sleep(1)
    
    # Set search parameters
    page.click("text=Shuffle Keywords")
    page.click("text=Run in background")
    
    # Start search
    page.click("text=Start Search")
    logger.info("Started search")
    
    # Wait for and verify results
    time.sleep(5)
    page.screenshot(path="manual_search_results.png")

@retry_on_error
def test_email_campaigns(page):
    """Test the Email Campaigns page functionality"""
    logger.info("Testing Email Campaigns page...")
    
    page.click('text="Email Campaigns"')
    page.wait_for_selector("h1:has-text('Automation Control Panel')", timeout=10000)
    
    # Test automation controls
    page.click('text="Start Automation"')
    time.sleep(2)
    page.click('text="Stop Automation"')
    
    # Test log controls
    page.click('text="Clear Logs"')
    
    page.screenshot(path="email_campaigns.png")

@retry_on_error
def test_projects_campaigns(page):
    """Test the Projects & Campaigns page functionality"""
    logger.info("Testing Projects & Campaigns page...")
    
    page.click('text="Projects & Campaigns"')
    page.wait_for_selector("h1:has-text('Projects & Campaigns')", timeout=10000)
    
    # Wait for content to load
    time.sleep(2)
    page.screenshot(path="projects_campaigns.png")

@retry_on_error
def test_knowledge_base(page):
    """Test the Knowledge Base page functionality"""
    logger.info("Testing Knowledge Base page...")
    
    page.click('text="Knowledge Base"')
    page.wait_for_selector("h1:has-text('Knowledge Base')", timeout=10000)
    
    # Fill in knowledge base form
    page.fill('input:below(:text("Knowledge Base Name"))', "Test KB")
    page.fill('textarea:below(:text("Bio"))', "Test Bio")
    page.fill('textarea:below(:text("Values"))', "Test Values")
    
    page.screenshot(path="knowledge_base.png")

@retry_on_error
def test_autoclient_ai(page):
    """Test the AutoClient AI page functionality"""
    logger.info("Testing AutoClient AI page...")
    
    page.click('text="AutoclientAI"')
    page.wait_for_selector("h1:has-text('AutoClient AI')", timeout=10000)
    
    # Test AI settings
    page.click('text="AI Model"')
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    
    page.screenshot(path="autoclient_ai.png")

def run(playwright):
    """Main test runner"""
    # Wait for Streamlit to be ready
    if not wait_for_streamlit():
        logger.error("Streamlit failed to start")
        sys.exit(1)

    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(
        viewport={'width': 1920, 'height': 1080},
        record_video_dir="videos/"
    )
    page = context.new_page()
    
    try:
        # Navigate to the Streamlit app
        page.goto("http://localhost:8501", wait_until="networkidle")
        logger.info("Navigated to Streamlit app")
        
        # Wait for initial load with increased timeout
        if not wait_for_element(page, "[data-testid='stSidebar']", timeout=60000):
            raise Exception("Streamlit sidebar failed to load")
        
        # Additional wait for app to stabilize
        time.sleep(10)
        
        # Run all tests
        test_manual_search(page)
        test_email_campaigns(page)
        test_projects_campaigns(page)
        test_knowledge_base(page)
        test_autoclient_ai(page)
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        page.screenshot(path="error_state.png")
        raise
    finally:
        context.close()
        browser.close()

if __name__ == "__main__":
    with sync_playwright() as playwright:
        run(playwright)
