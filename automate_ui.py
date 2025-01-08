<<<<<<< HEAD
from playwright.sync_api import sync_playwright
=======
from playwright.sync_api import sync_playwright, expect
>>>>>>> a37c652 (refactor: enhance automation system and UI/UX improvements)
import time

def automate_search():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
<<<<<<< HEAD
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
=======
        # Go to the Streamlit app and wait for load
        page.goto('http://localhost:8505')
        page.wait_for_load_state('networkidle')
        time.sleep(5)  # Extra wait for Streamlit
        
        # Click Manual Search in sidebar
        page.locator('div[data-testid="stSidebarNav"] >> text=Manual Search').click()
        time.sleep(3)
        
        # Enable email sending if not enabled
        email_checkbox = page.locator('label:has-text("Enable email sending") input[type="checkbox"]')
        if not email_checkbox.is_checked():
            email_checkbox.check()
        time.sleep(2)
        
        # Select first email template
        page.locator('div:has-text("Email template") >> select').click()
        page.keyboard.press('ArrowDown')
        page.keyboard.press('Enter')
        time.sleep(2)
        
        # Select first email setting
        page.locator('div:has-text("From Email") >> select').click()
        page.keyboard.press('ArrowDown')
        page.keyboard.press('Enter')
        time.sleep(2)
        
        # Set number of results to 10
        slider = page.locator('div[data-testid="stSlider"]')
        slider.click()
        page.keyboard.type('10')
        page.keyboard.press('Enter')
        time.sleep(2)

        # Add search terms
        search_terms_input = page.locator('div[data-testid="stTextInput"] input').first
        search_terms_input.click()
        search_terms_input.type('agencia marketing digital barcelona')
        page.keyboard.press('Enter')
        time.sleep(1)
        
        search_terms_input.type('agencia diseño web madrid')
        page.keyboard.press('Enter')
        time.sleep(1)
        
        # Click Search button - using more specific selector and force click
        search_button = page.locator('button:has-text("Search")').last
        search_button.click(force=True)
        print("Search started... Monitoring for leads")
        
        try:
            while True:
                # Monitor for leads and emails
                content = page.content()
                if "🟢 Saved lead:" in content:
                    leads = page.locator('text=/🟢 Saved lead:.*/').all_text_contents()
                    for lead in leads:
                        print(lead.strip())
                if "🟣 Sent email to:" in content:
                    emails = page.locator('text=/🟣 Sent email to:.*/').all_text_contents()
                    for email in emails:
                        print(email.strip())
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nStopping automation...")
        finally:
            browser.close()
>>>>>>> a37c652 (refactor: enhance automation system and UI/UX improvements)

if __name__ == "__main__":
    automate_search() 