import unittest
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

class StreamlitAppTest(unittest.TestCase):
    def setUp(self):
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()
        # Local Streamlit app URL
        self.url = "http://localhost:8501"
        self.wait = WebDriverWait(self.driver, 10)
    
    def tearDown(self):
        if self.driver:
            self.driver.quit()
    
    def test_manual_search_page(self):
        """Test the manual search functionality"""
        try:
            # Navigate to app
            self.driver.get(self.url)
            sleep(5)  # Wait for Streamlit to load
            
            # Verify page title
            self.assertIn("Email Lead Generator", self.driver.title)
            
            # Find and click Manual Search in navigation
            nav_items = self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='stVerticalBlock'] .nav-link"))
            )
            manual_search = next(item for item in nav_items if "Manual Search" in item.text)
            manual_search.click()
            sleep(2)
            
            # Enter search term
            search_input = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[aria-label='Enter search terms:']"))
            )
            search_input.send_keys("software engineer")
            search_input.send_keys("\n")
            
            # Click search button
            search_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Start Search')]"))
            )
            search_button.click()
            
            # Wait for results
            sleep(10)
            
            # Verify results appeared
            results_container = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='stExpander']"))
            )
            self.assertTrue(results_container.is_displayed())
            
            print("Manual search test passed!")
            
        except TimeoutException as e:
            self.fail(f"Timeout waiting for element: {str(e)}")
        except Exception as e:
            self.fail(f"Test failed: {str(e)}")
    
    def test_email_campaigns_page(self):
        """Test the email campaigns page"""
        try:
            self.driver.get(self.url)
            sleep(5)
            
            # Navigate to Email Campaigns
            nav_items = self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='stVerticalBlock'] .nav-link"))
            )
            campaigns_link = next(item for item in nav_items if "Email Campaigns" in item.text)
            campaigns_link.click()
            sleep(2)
            
            # Verify metrics are displayed
            metrics = self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "[data-testid='metric-container']"))
            )
            self.assertGreater(len(metrics), 0)
            
            print("Email campaigns test passed!")
            
        except TimeoutException as e:
            self.fail(f"Timeout waiting for element: {str(e)}")
        except Exception as e:
            self.fail(f"Test failed: {str(e)}")

def debug_ui():
    """Run UI tests with debugging enabled"""
    try:
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(StreamlitAppTest)
        
        # Run tests with detailed output
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Generate debug report
        with open('debug_report.txt', 'w') as f:
            f.write(f"Tests run: {result.testsRun}\n")
            f.write(f"Failures: {len(result.failures)}\n")
            f.write(f"Errors: {len(result.errors)}\n")
            
            if result.failures:
                f.write("\nFailures:\n")
                for failure in result.failures:
                    f.write(f"{failure[0]}: {failure[1]}\n")
            
            if result.errors:
                f.write("\nErrors:\n")
                for error in result.errors:
                    f.write(f"{error[0]}: {error[1]}\n")
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"Error running tests: {str(e)}")
        return False

if __name__ == '__main__':
    debug_ui()