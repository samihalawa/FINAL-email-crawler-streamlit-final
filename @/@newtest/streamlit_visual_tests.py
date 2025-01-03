import pytest
import asyncio
from playwright.sync_api import Page, expect, Browser
from datetime import datetime, timedelta
import time
import logging
from typing import Generator, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    "base_url": "http://localhost:8501",
    "screenshot_dir": "test_results/visual",
    "timeout": 30000,
    "retry_attempts": 3,
    "retry_delay": 1000,
}

class TestHelper:
    """Helper class for common test operations"""
    
    @staticmethod
    async def wait_for_selector_safe(page: Page, selector: str, timeout: int = 5000) -> bool:
        """Safely wait for a selector with timeout and error handling"""
        try:
            await page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception as e:
            logger.warning(f"Selector '{selector}' not found: {str(e)}")
            return False

    @staticmethod
    async def retry_action(action: callable, max_attempts: int = 3, delay: int = 1000):
        """Retry an action with exponential backoff"""
        for attempt in range(max_attempts):
            try:
                return await action()
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                wait_time = delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}ms...")
                await asyncio.sleep(wait_time / 1000)

    @staticmethod
    def ensure_screenshot_dir(path: str):
        """Ensure screenshot directory exists"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

@pytest.mark.visual
class TestStreamlitVisual:
    
    @pytest.fixture(autouse=True)
    async def setup(self, page: Page):
        """Setup test environment with improved error handling"""
        self.helper = TestHelper()
        try:
            await page.goto(TEST_CONFIG["base_url"])
            assert await self.helper.wait_for_selector_safe(page, "[data-testid='stAppHeader']")
            self.helper.ensure_screenshot_dir(f"{TEST_CONFIG['screenshot_dir']}/setup.png")
            await page.screenshot(path=f"{TEST_CONFIG['screenshot_dir']}/setup.png")
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            raise

    async def test_01_form_persistence(self, page: Page):
        """Test form persistence with improved resilience"""
        async def fill_form():
            await page.fill("[data-testid='search-terms-input']", "test term")
            await page.click("[data-testid='ignore-fetched-checkbox']")
            
        await self.helper.retry_action(fill_form)
        
        # Take screenshots with error handling
        try:
            self.helper.ensure_screenshot_dir(f"{TEST_CONFIG['screenshot_dir']}/01_form_filled.png")
            await page.screenshot(path=f"{TEST_CONFIG['screenshot_dir']}/01_form_filled.png")
            
            await page.reload()
            await asyncio.sleep(2)  # Allow for state rehydration
            
            # Verify persistence with flexible selectors
            input_value = await page.input_value("[data-testid='search-terms-input']")
            assert input_value == "test term", f"Expected 'test term', got '{input_value}'"
            
            checkbox_state = await page.is_checked("[data-testid='ignore-fetched-checkbox']")
            assert checkbox_state, "Checkbox state not persisted"
            
        except Exception as e:
            logger.error(f"Form persistence test failed: {str(e)}")
            raise

    async def test_02_background_process(self, page: Page, browser: Browser):
        """Test background process continuation with improved monitoring"""
        process_id = None
        
        try:
            # Start process
            await page.fill("[data-testid='search-terms-input']", "background test")
            await page.click("[data-testid='start-search-button']")
            
            # Capture process ID
            process_element = await page.wait_for_selector("[data-testid='process-container']")
            process_id = await process_element.get_attribute("data-process-id")
            
            # Close browser
            await browser.close()
            
            # Wait and verify continuation
            await asyncio.sleep(30)
            
            # Reopen and verify
            new_page = await browser.new_page()
            await new_page.goto(TEST_CONFIG["base_url"])
            
            # Wait for process container with retry
            async def verify_process():
                container = await new_page.wait_for_selector(
                    f"[data-testid='process-container'][data-process-id='{process_id}']"
                )
                return container is not None
            
            assert await self.helper.retry_action(verify_process)
            
        except Exception as e:
            logger.error(f"Background process test failed: {str(e)}")
            if process_id:
                logger.error(f"Failed process ID: {process_id}")
            raise

    async def test_03_real_time_updates(self, page: Page):
        """Test real-time updates with improved monitoring"""
        try:
            # Start monitoring logs
            log_monitor = await page.wait_for_selector("[data-testid='log-container']")
            initial_logs = await log_monitor.text_content()
            
            # Start process
            await page.click("[data-testid='start-search-button']")
            
            # Monitor for updates with timeout
            async def wait_for_log_update():
                updated_logs = await log_monitor.text_content()
                return updated_logs != initial_logs
            
            assert await self.helper.retry_action(wait_for_log_update)
            
            # Verify scroll behavior
            scroll_position = await page.evaluate("""
                document.querySelector('[data-testid="log-container"]').scrollTop
            """)
            assert scroll_position > 0, "Log container did not auto-scroll"
            
        except Exception as e:
            logger.error(f"Real-time update test failed: {str(e)}")
            raise

    # Additional test methods...

    async def test_20_cross_device_sync(self, page: Page, browser: Browser):
        """Test cross-device synchronization with improved verification"""
        try:
            # Start process on first device
            await page.fill("[data-testid='search-terms-input']", "sync test")
            await page.click("[data-testid='start-search-button']")
            
            # Get initial state
            initial_state = await page.evaluate("""
                () => ({
                    logs: document.querySelector('[data-testid="log-container"]').textContent,
                    metrics: document.querySelector('[data-testid="metrics-container"]').textContent
                })
            """)
            
            # Simulate second device
            page2 = await browser.new_page()
            await page2.goto(TEST_CONFIG["base_url"])
            
            # Verify sync with retry
            async def verify_sync():
                new_state = await page2.evaluate("""
                    () => ({
                        logs: document.querySelector('[data-testid="log-container"]').textContent,
                        metrics: document.querySelector('[data-testid="metrics-container"]').textContent
                    })
                """)
                return (
                    new_state["logs"] == initial_state["logs"] and
                    new_state["metrics"] == initial_state["metrics"]
                )
            
            assert await self.helper.retry_action(verify_sync)
            
        except Exception as e:
            logger.error(f"Cross-device sync test failed: {str(e)}")
            raise

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--headed"]) 