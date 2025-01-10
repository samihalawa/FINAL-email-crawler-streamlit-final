import requests
import json
import time
from RELEASE_streamlit_app import db_session, manual_search, save_lead, bulk_send_emails
from RELEASE_streamlit_app import get_active_project_id, SearchTerm, EmailTemplate

def simulate_manual_search_clicks():
    print("Simulating Manual Search page clicks...")
    
    with db_session() as session:
        # 1. Add test search term
        test_term = SearchTerm(
            term="test search term",
            campaign_id=get_active_project_id(),
            created_at=time.time()
        )
        session.add(test_term)
        session.commit()
        print("Added test search term")
        
        # 2. Perform manual search
        results = manual_search(
            session,
            ["test search term"],
            num_results=10,
            ignore_previously_fetched=True
        )
        print(f"Search results: {results}")
        
        # 3. Save any found leads
        if results and 'results' in results:
            for res in results['results']:
                lead = save_lead(
                    session,
                    res['Email'],
                    url=res.get('URL')
                )
                print(f"Saved lead: {lead.email if lead else 'Failed'}")
        
        # 4. Test email template
        template = EmailTemplate(
            template_name="Test Template",
            subject="Test Subject",
            body_content="Test body content",
            campaign_id=get_active_project_id(),
            project_id=get_active_project_id()
        )
        session.add(template)
        session.commit()
        print("Added test email template")
        
        # 5. Test bulk email sending
        if results and 'results' in results:
            logs, sent_count = bulk_send_emails(
                session,
                template.id,
                "test@example.com",
                "reply@example.com",
                [{'Email': res['Email']} for res in results['results']]
            )
            print(f"Sent {sent_count} test emails")

if __name__ == "__main__":
    simulate_manual_search_clicks() 