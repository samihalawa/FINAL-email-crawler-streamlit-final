from playwright.sync_api import sync_playwright
import time

def automate_search():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        # Navigate to the Streamlit app
        page.goto("http://localhost:8503")
        time.sleep(3)  # Wait for load
        
        # Click on Manual Search in sidebar
        page.click("text=🔍 Manual Search")
        time.sleep(2)
        
        # Best search terms for telefonillo campaign
        search_terms = [
            "gestores pisos turisticos madrid",
            "agencia airbnb barcelona llaves",
            "empresa gestion apartamentos turisticos valencia",
            "property manager malaga airbnb",
            "administrador fincas portero automatico",
            "modernizar telefonillo edificio",
            "comunidad vecinos telefonillo antiguo"
        ]
        
        # Enter search terms
        for term in search_terms:
            page.keyboard.type(term)
            page.keyboard.press("Enter")
            time.sleep(0.5)
        
        # Set results to 10000
        page.fill('input[aria-label="Results per term"]', "10000")
        
        # Enable email sending
        page.click('text="Enable email sending"')
        time.sleep(1)
        
        # Select template
        page.click('select[aria-label="Email template"]')
        page.click('text="Control Remoto Telefonillo sin Permiso de Comunidad ni Obras 🔥 🚪"')
        
        # Select From Email (Housemoney | Sami Halawa)
        page.click('select[aria-label="From Email"]')
        page.click('text="Housemoney | Sami Halawa"')
        
        # Click Search button
        page.click('text="Search"')
        
        # Keep browser open to monitor progress
        print("Search started! Monitoring progress...")
        page.wait_for_timeout(999999)  # Keep running until manually closed

if __name__ == "__main__":
    automate_search() 