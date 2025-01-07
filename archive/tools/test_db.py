from RELEASE_streamlit_app import db_session, Project, Campaign

with db_session() as session:
    existing_project = session.query(Project).first()
    if existing_project:
        print(f"Using existing project (ID: {existing_project.id})")
        project = existing_project
    else:
        project = Project(project_name="Test Project")
        session.add(project)
        session.flush()
        print(f"Created new project (ID: {project.id})")

    campaign = Campaign(project_id=project.id, campaign_name="Test Campaign")
    session.add(campaign)
    session.commit()
    print(f"Created campaign (ID: {campaign.id})")
