import sys
import logging
from contextlib import redirect_stdout, redirect_stderr
import io
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Capture output
output = io.StringIO()
error = io.StringIO()

try:
    with redirect_stdout(output), redirect_stderr(error):
        # Import and test key functions
        from streamlit_app import (
            db_session, 
            safe_db_session,
            fetch_email_templates,
            save_email_campaign,
            bulk_send_emails,
            manual_search,
            run_automation_cycle
        )
        
        # Test imports completed
        print("✓ Imports successful")
        
        # Test database session
        with db_session() as session:
            # Test template fetching
            templates = fetch_email_templates(session)
            print(f"✓ Found {len(templates)} email templates")
            
            # Test template access
            if templates:
                template_id = int(templates[0].split(':')[0])
                print(f"✓ Successfully parsed template ID: {template_id}")
                
except Exception as e:
    print("\n❌ Error occurred:")
    print(f"Type: {type(e).__name__}")
    print(f"Message: {str(e)}")
    print("\nTraceback:")
    traceback.print_exc()
    
finally:
    # Print captured output
    print("\nStandard Output:")
    print(output.getvalue())
    print("\nError Output:")
    print(error.getvalue())