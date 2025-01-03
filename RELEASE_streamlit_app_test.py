import pytest
from playwright.sync_api import Page, expect
from datetime import datetime, timedelta
import time

# Test Suite for Manual Search and Background Processing

@pytest.mark.visual
class TestManualSearchVisual:
    
    @pytest.fixture(autouse=True)
    async def setup(self, page: Page):
        await page.goto("http://localhost:8501")
        # Wait for Streamlit to fully load
        await page.wait_for_selector("[data-testid='stAppHeader']")
        
    async def test_01_manual_search_form_persistence(self, page: Page):
        """Test that form values persist across browser sessions"""
        # Take screenshot of initial state
        await page.screenshot(path="test_results/01_initial_state.png")
        
        # Fill form with test data
        await page.fill("[data-testid='search-terms-input']", "test term")
        await page.click("[data-testid='ignore-fetched-checkbox']")
        
        # Take screenshot of filled form
        await page.screenshot(path="test_results/01_filled_form.png")
        
        # Refresh page
        await page.reload()
        
        # Verify form values persisted
        expect(await page.input_value("[data-testid='search-terms-input']")).toBe("test term")
        expect(await page.is_checked("[data-testid='ignore-fetched-checkbox']")).toBeTruthy()
        
        # Take screenshot of persisted state
        await page.screenshot(path="test_results/01_persisted_state.png")

    async def test_02_background_process_continuation(self, page: Page):
        """Test that background processes continue after browser close"""
        # Start a search process
        await page.fill("[data-testid='search-terms-input']", "background test")
        await page.click("[data-testid='start-search-button']")
        
        # Take screenshot of process start
        await page.screenshot(path="test_results/02_process_start.png")
        
        # Close browser
        await page.close()
        
        # Wait 30 seconds
        time.sleep(30)
        
        # Reopen browser and navigate back
        page = await browser.new_page()
        await page.goto("http://localhost:8501")
        
        # Verify process continued
        logs = await page.text_content("[data-testid='process-logs']")
        expect(logs).toContain("Process continued during browser closure")
        
        # Take screenshot of continued process
        await page.screenshot(path="test_results/02_process_continued.png")

    async def test_03_real_time_log_updates(self, page: Page):
        """Test that logs update in real-time"""
        # Start process
        await page.click("[data-testid='start-search-button']")
        
        # Take initial log screenshot
        await page.screenshot(path="test_results/03_initial_logs.png")
        
        # Wait for new logs
        await page.wait_for_selector("text=New lead found")
        
        # Take updated log screenshot
        await page.screenshot(path="test_results/03_updated_logs.png")
        
        # Verify log scroll position
        scroll_position = await page.evaluate("document.querySelector('[data-testid=\"log-container\"]').scrollTop")
        expect(scroll_position).toBeGreaterThan(0)

    async def test_04_email_sending_visualization(self, page: Page):
        """Test email sending animation and status updates"""
        # Setup email process
        await page.click("[data-testid='enable-email-checkbox']")
        await page.fill("[data-testid='search-terms-input']", "email test")
        
        # Take pre-send screenshot
        await page.screenshot(path="test_results/04_pre_send.png")
        
        # Start process
        await page.click("[data-testid='start-search-button']")
        
        # Wait for email animation
        await page.wait_for_selector("[data-testid='email-animation']")
        
        # Take animation screenshot
        await page.screenshot(path="test_results/04_email_animation.png")
        
        # Verify email status updates
        expect(await page.is_visible("[data-testid='email-success-icon']")).toBeTruthy()

    async def test_05_process_control_interface(self, page: Page):
        """Test process control UI elements"""
        # Take initial controls screenshot
        await page.screenshot(path="test_results/05_initial_controls.png")
        
        # Start process
        await page.click("[data-testid='start-search-button']")
        
        # Take active process screenshot
        await page.screenshot(path="test_results/05_active_process.png")
        
        # Pause process
        await page.click("[data-testid='pause-button']")
        
        # Take paused state screenshot
        await page.screenshot(path="test_results/05_paused_state.png")
        
        # Verify control states
        expect(await page.is_enabled("[data-testid='resume-button']")).toBeTruthy()
        expect(await page.is_disabled("[data-testid='pause-button']")).toBeTruthy()

    # Additional tests...
    
    async def test_15_multi_process_management(self, page: Page):
        """Test managing multiple concurrent processes"""
        # Start multiple processes
        for i in range(3):
            await page.fill("[data-testid='search-terms-input']", f"test {i}")
            await page.click("[data-testid='start-search-button']")
            
        # Take multi-process screenshot
        await page.screenshot(path="test_results/15_multi_process.png")
        
        # Verify process isolation
        process_containers = await page.query_selector_all("[data-testid='process-container']")
        expect(len(process_containers)).toBe(3)
        
        # Verify independent log streams
        for container in process_containers:
            logs = await container.text_content()
            expect(logs).toContain(f"Process {container.get_attribute('data-process-id')}")

    async def test_16_persistence_across_devices(self, page: Page):
        """Test application state persistence across different devices"""
        # Set up test data
        await page.fill("[data-testid='search-terms-input']", "cross device test")
        await page.click("[data-testid='start-search-button']")
        
        # Take initial device screenshot
        await page.screenshot(path="test_results/16_device1_state.png")
        
        # Simulate different device/browser
        page2 = await browser.new_page()
        await page2.goto("http://localhost:8501")
        
        # Take second device screenshot
        await page2.screenshot(path="test_results/16_device2_state.png")
        
        # Verify state synchronization
        expect(await page2.text_content("[data-testid='process-logs']")).toEqual(
            await page.text_content("[data-testid='process-logs']")
        )

    async def test_17_error_recovery_visualization(self, page: Page):
        """Test error handling and recovery visualization"""
        # Trigger error condition
        await page.evaluate("window.triggerTestError()")
        
        # Take error state screenshot
        await page.screenshot(path="test_results/17_error_state.png")
        
        # Verify error visualization
        expect(await page.is_visible("[data-testid='error-notification']")).toBeTruthy()
        
        # Test recovery
        await page.click("[data-testid='retry-button']")
        
        # Take recovery screenshot
        await page.screenshot(path="test_results/17_recovered_state.png")

    async def test_18_real_time_metrics_update(self, page: Page):
        """Test real-time metrics visualization"""
        # Start process
        await page.click("[data-testid='start-search-button']")
        
        # Take initial metrics screenshot
        await page.screenshot(path="test_results/18_initial_metrics.png")
        
        # Wait for metrics update
        await page.wait_for_function("""
            () => document.querySelector('[data-testid="leads-found-metric"]').textContent !== "0"
        """)
        
        # Take updated metrics screenshot
        await page.screenshot(path="test_results/18_updated_metrics.png")

    async def test_19_responsive_layout(self, page: Page):
        """Test responsive layout across different screen sizes"""
        # Test desktop layout
        await page.set_viewport_size({"width": 1920, "height": 1080})
        await page.screenshot(path="test_results/19_desktop_layout.png")
        
        # Test tablet layout
        await page.set_viewport_size({"width": 768, "height": 1024})
        await page.screenshot(path="test_results/19_tablet_layout.png")
        
        # Test mobile layout
        await page.set_viewport_size({"width": 375, "height": 812})
        await page.screenshot(path="test_results/19_mobile_layout.png")

    async def test_20_theme_persistence(self, page: Page):
        """Test theme persistence and visualization"""
        # Switch to dark theme
        await page.click("[data-testid='theme-toggle']")
        
        # Take dark theme screenshot
        await page.screenshot(path="test_results/20_dark_theme.png")
        
        # Refresh page
        await page.reload()
        
        # Verify theme persisted
        expect(await page.evaluate("document.documentElement.getAttribute('data-theme')")).toBe("dark")
        
        # Take persisted theme screenshot
        await page.screenshot(path="test_results/20_theme_persisted.png")