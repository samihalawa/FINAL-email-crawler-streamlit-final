from playwright.sync_api import sync_playwright
import time
import os

def run_search_and_analyze():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        try:
            # Navigate and perform search
            print("Starting search automation...")
            page.goto("http://localhost:8501")
            time.sleep(2)
            
            # Click Manual Search
            page.click("text=Manual Search")
            time.sleep(1)
            
            # Enter search terms
            search_terms = ["software developers spain", 
                          "tech companies barcelona",
                          "startup jobs madrid"]
            
            for term in search_terms:
                print(f"\nSearching for: {term}")
                
                # Clear previous input if any
                page.evaluate('document.querySelector("input").value = ""')
                page.fill("input", term)
                time.sleep(1)
                
                # Click search and wait
                page.click("text=Search")
                time.sleep(5)
                
                # Take screenshot
                screenshot_name = f"search_results_{term.replace(' ', '_')}.png"
                page.screenshot(path=screenshot_name)
                print(f"Screenshot saved as: {screenshot_name}")
                
                # Extract and log results
                results = page.evaluate('''() => {
                    const results = [];
                    document.querySelectorAll('.stMarkdown').forEach(el => {
                        results.push(el.innerText);
                    });
                    return results;
                }''')
                
                print(f"Found {len(results)} results")
                
                # Wait before next search
                time.sleep(2)
            
        except Exception as e:
            print(f"Error during automation: {str(e)}")
        finally:
            browser.close()

if __name__ == "__main__":
    print("Starting search and analysis process...")
    run_search_and_analyze() 