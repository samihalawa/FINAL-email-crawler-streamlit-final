from streamlit_app import db_session, EmailTemplate, EmailCampaign

def delete_templates():
    with db_session() as session:
        try:
            # First, get the template we want to keep
            keep_template = session.query(EmailTemplate).filter(
                EmailTemplate.subject.ilike('%telefonillo%')
            ).first()
            
            if not keep_template:
                print("No template found with 'telefonillo' in subject")
                return
            
            # Delete email campaigns for templates we want to remove
            deleted_campaigns = session.query(EmailCampaign).filter(
                EmailCampaign.template_id != keep_template.id
            ).delete(synchronize_session=False)
            
            # Now delete the templates
            deleted_templates = session.query(EmailTemplate).filter(
                EmailTemplate.id != keep_template.id
            ).delete(synchronize_session=False)
            
            session.commit()
            
            print(f"Deleted {deleted_campaigns} email campaigns")
            print(f"Deleted {deleted_templates} templates")
            print("\nRemaining template:")
            print(f"ID: {keep_template.id}")
            print(f"Name: {keep_template.template_name}")
            print(f"Subject: {keep_template.subject}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            session.rollback()

if __name__ == "__main__":
    delete_templates() 