def bulk_send_emails(session, template_id, from_email, reply_to, leads, **kwargs):
    progress_bar = kwargs.get('progress_bar')
    status_text = kwargs.get('status_text')
    log_container = kwargs.get('log_container')
    results = kwargs.get('results', [])
    
    sent_count = 0≤
    total = len(leads)
    
    template = session.query(EmailTemplate).get(template_id)
    if not template:
        raise ValueError("Email template not found")
    
    try:
        for idx, lead in enumerate(leads):
            try:
                response, tracking_id = rate_limited_send_email(
                    session, 
                    from_email, 
                    lead['Email'], 
                    template.subject, 
                    template.body_content, 
                    reply_to=reply_to
                )
                status = 'sent' if response else 'failed'
                results.append({'Email': lead['Email'], 'Status': status})
                sent_count += 1 if status == 'sent' else 0
                if log_container:
                    log_container.text(f"Sent to {lead['Email']}: {status}")
                
                if status == 'sent':
                    save_email_campaign(
                        session, 
                        lead['Email'], 
                        template_id, 
                        'Sent', 
                        datetime.utcnow(), 
                        template.subject, 
                        response['MessageId'], 
                        template.body_content
                    )
            except Exception as e:
                results.append({'Email': lead['Email'], 'Status': 'error'})
                if log_container:
                    log_container.error(f"Error sending to {lead['Email']}: {str(e)}")
            
            if progress_bar:
                progress = (idx + 1) / total
                progress_bar.progress(progress)
            if status_text:
                status_text.text(f"Sending emails: {int(progress * 100)}% completed.")
    except Exception as e:
        if log_container:
            log_container.error(f"Error in bulk email sending: {str(e)}")
    finally:
        return {"logs": results, "sent_count": sent_count}

def manual_search_page():
    st.title("Manual Search")

    start_background_process()

    user_settings = get_user_settings()

    with safe_db_session() as session:
        try:
            state = get_background_state()
            if state:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Background Status", "Running" if state['is_running'] else "Paused")
                with col2:
                    st.metric("Total Leads Found", state['leads_found'])
                with col3:
                    st.metric("Total Emails Sent", state['emails_sent'])

                st.info(f"Last run: {state['last_run'] or 'Never'} | Current term: {state['current_term'] or 'None'}")

            latest_leads = session.query(Lead).order_by(Lead.created_at.desc()).limit(5).all()
            latest_leads_data = [(lead.email, lead.company, lead.created_at) for lead in latest_leads]

            latest_campaigns = session.query(EmailCampaign).order_by(EmailCampaign.sent_at.desc()).limit(5).all()
            latest_campaigns_data = [(campaign.lead.email, campaign.template.template_name, campaign.sent_at, campaign.status) 
                                     for campaign in latest_campaigns]

            recent_searches = session.query(SearchTerm).order_by(SearchTerm.created_at.desc()).limit(5).all()
            recent_search_terms = [term.term for term in recent_searches]

            # ... (rest of the function remains unchanged)

        except Exception as e:
            st.error(f"An error occurred while loading data: {str(e)}")
            log_error(f"Error in manual_search_page: {str(e)}")

def autoclient_ai_page():
    st.header("AutoclientAI - Automated Lead Generation")
    with st.expander("Knowledge Base Information", expanded=False):
        with safe_db_session() as session:
            kb_info = get_knowledge_base_info(session, get_active_project_id())
        if not kb_info:
            return st.error("Knowledge Base not found for the active project. Please set it up first.")
        st.json(kb_info)
    user_input = st.text_area("Enter additional context or specific goals for lead generation:", help="This information will be used to generate more targeted search terms.")
    if st.button("Generate Optimized Search Terms", key="generate_optimized_terms"):
        with st.spinner("Generating optimized search terms..."):
            try:
                with safe_db_session() as session:
                    base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
                    optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
                if optimized_terms:
                    st.session_state.optimized_terms = optimized_terms
                    st.success("Search terms optimized successfully!")
                    st.subheader("Optimized Search Terms")
                    st.write(", ".join(optimized_terms))
                else:
                    st.warning("No optimized search terms were generated. Please try again or adjust your input.")
            except Exception as e:
                st.error(f"An error occurred while generating optimized search terms: {str(e)}")
                log_error(f"Error in generate_optimized_search_terms: {str(e)}")

    # ... (rest of the function remains unchanged)

def projects_campaigns_page():
    with safe_db_session() as session:
        st.header("Projects and Campaigns")
        st.subheader("Add New Project")
        with st.form("add_project_form"):
            project_name = st.text_input("Project Name")
            if st.form_submit_button("Add Project"):
                if project_name.strip():
                    try:
                        new_project = Project(project_name=project_name, created_at=datetime.utcnow())
                        session.add(new_project)
                        session.commit()
                        st.success(f"Project '{project_name}' added successfully.")
                    except SQLAlchemyError as e:
                        session.rollback()
                        st.error(f"Error adding project: {str(e)}")
                else:
                    st.warning("Please enter a project name.")
        
        try:
            projects = session.query(Project).all()
            for project in projects:
                with st.expander(f"{project.project_name} (ID: {project.id})"):
                    campaigns = session.query(Campaign).filter_by(project_id=project.id).all()
                    for campaign in campaigns:
                        with st.expander(f"{campaign.campaign_name} (ID: {campaign.id})"):
                            st.write(f"Created At: {campaign.created_at}")
                            st.write(f"Last Run: {campaign.last_run or 'Never'}")
                            st.write(f"Total Leads Found: {campaign.total_leads_found}")
                            st.write(f"Total Emails Sent: {campaign.total_emails_sent}")
                            if st.button(f"Delete Campaign {campaign.id}", key=f"delete_campaign_{campaign.id}"):
                                try:
                                    session.delete(campaign)
                                    session.commit()
                                    st.success(f"Campaign {campaign.id} deleted successfully.")
                                    st.experimental_rerun()
                                except SQLAlchemyError as e:
                                    session.rollback()
                                    st.error(f"Error deleting campaign: {str(e)}")
                    
                    # ... (rest of the function remains unchanged)

        except SQLAlchemyError as e:
            st.error(f"An error occurred while loading projects and campaigns: {str(e)}")
            log_error(f"Error in projects_campaigns_page: {str(e)}")

def knowledge_base_page():
    st.title("Knowledge Base")
    with safe_db_session() as session:
        try:
            project_options = fetch_projects(session)
            if not project_options:
                return st.warning("No projects found. Please create a project first.")
            selected_project = st.selectbox("Select Project", options=project_options)
            project_id = int(selected_project.split(":")[0])
            set_active_project_id(project_id)
            kb_entry = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
            with st.form("knowledge_base_form"):
                fields = ['kb_name', 'kb_bio', 'kb_values', 'contact_name', 'contact_role', 'contact_email', 'company_description', 'company_mission', 'company_target_market', 'company_other', 'product_name', 'product_description', 'product_target_customer', 'product_other', 'other_context', 'example_email']
                form_data = {field: st.text_input(field.replace('_', ' ').title(), value=getattr(kb_entry, field, '')) if field in ['kb_name', 'contact_name', 'contact_role', 'contact_email', 'product_name'] else st.text_area(field.replace('_', ' ').title(), value=getattr(kb_entry, field, '')) for field in fields}
                if st.form_submit_button("Save Knowledge Base"):
                    try:
                        form_data.update({'project_id': project_id, 'created_at': datetime.utcnow()})
                        if kb_entry:
                            for k, v in form_data.items():
                                setattr(kb_entry, k, v)
                        else:
                            session.add(KnowledgeBase(**form_data))
                        session.commit()
                        st.success("Knowledge Base saved successfully!", icon="✅")
                    except SQLAlchemyError as e:
                        session.rollback()
                        st.error(f"An error occurred while saving the Knowledge Base: {str(e)}")
                        log_error(f"Error saving Knowledge Base: {str(e)}")
        except Exception as e:
            st.error(f"An error occurred while loading the Knowledge Base page: {str(e)}")
            log_error(f"Error in knowledge_base_page: {str(e)}")
