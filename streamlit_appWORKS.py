        session.rollback()
        logging.error(f"Database error: {str(e)}")
        raise

def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    try:
        lead = session.query(Lead).filter_by(email=lead_email).first()
        if not lead:
            logging.error(f"Lead with email {lead_email} not found.")
            return

        new_campaign = EmailCampaign(
            lead_id=lead.id,
            template_id=template_id,
            status=status,
            sent_at=sent_at,
            customized_subject=subject or "No subject",
            message_id=message_id or f"unknown-{uuid.uuid4()}",
            customized_content=email_body or "No content",
            campaign_id=get_active_campaign_id(),
            tracking_id=str(uuid.uuid4())
        )
        session.add(new_campaign)
        session.commit()
    except Exception as e:
        logging.error(f"Error saving email campaign: {str(e)}")
        session.rollback()

def update_log(log_container, message, level='info'):
    icon = {'info': '🔵', 'success': '🟢', 'warning': '🟠', 'error': '🔴', 'email_sent': '🟣'}.get(level, '⚪')
    log_entry = f"{icon} {message}"

    # Log to file or console without HTML
    logging.log(getattr(logging, level.upper(), logging.INFO), message)

    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []

    st.session_state.log_entries.append(log_entry)

    if log_container is not None:
        log_container.markdown("\n".join(st.session_state.log_entries))

def optimize_search_term(search_term, language):
    if language == 'english':
        return f'"{search_term}" email OR contact OR "get in touch" site:.com'
    elif language == 'spanish':
        return f'"{search_term}" correo OR contacto OR "ponte en contacto" site:.es'
    return search_term

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

def get_domain_from_url(url):
    return urlparse(url).netloc

def is_valid_email(email):
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

def extract_emails_from_html(html_content, domain):
    soup = BeautifulSoup(html_content, 'html.parser')
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, html_content)

    page_title = soup.title.string if soup.title else "No title found"
    meta_description = soup.find('meta', attrs={'name': 'description'})
    meta_description = meta_description['content'] if meta_description else "No description found"

    results = []
    for email in emails:
        name, company, job_title = extract_info_from_page(soup)
        results.append({
            'email': email,
            'name': name,
            'company': company,
            'job_title': job_title,
            'page_title': page_title,
            'meta_description': meta_description,
            'tags': [tag.name for tag in soup.find_all()],
            'domain': domain
        })

    return results

def extract_info_from_page(soup):
    name = soup.find('meta', {'name': 'author'})
    name = name['content'] if name else ''

    company = soup.find('meta', {'property': 'og:site_name'})
    company = company['content'] if company else ''

    job_title = soup.find('meta', {'name': 'job_title'})
    job_title = job_title['content'] if job_title else ''

    return name, company, job_title
@functools.lru_cache(maxsize=100)
def optimized_search(term, num_results):
    # Implement more efficient search algorithm here
    # This is a placeholder for demonstration
    return google_search(term, num_results)

async def async_fetch(url, session):
    try:
        async with session.get(url, timeout=10) as response:
            return await response.text()
    except Exception as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return None

async def manual_search(session, terms, num_results, **kwargs):
    results, total_leads = [], 0
    async with aiohttp.ClientSession() as client:
        for term in terms:
            search_results = await asyncio.gather(*[async_fetch(url, client) for url in optimized_search(term, num_results)])
            for html_content in search_results:
                if html_content:
                    emails = extract_emails_from_html(html_content, get_domain_from_url(url))
                    # Process emails and update results
                    # ... (rest of the code)
    return {"total_leads": total_leads, "results": results}

def delete_lead(session, lead_id):
    try:
        session.query(LeadSource).filter(LeadSource.lead_id == lead_id).delete()
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            session.delete(lead)
        session.commit()
        return True
    except SQLAlchemyError as e:
        logging.error(f"Error deleting lead {lead_id} and its sources: {str(e)}")
        session.rollback()
        return False

def background_manual_search(session_factory):
    with safe_db_session() as session:
        state = get_background_state()
        if not state['is_running']:
            return

        search_terms = session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()
        for term in search_terms:
            update_background_state(session, current_term=term.term)
            results = asyncio.run(manual_search(session, [term.term], 10, ignore_previously_fetched=True))
            update_background_state(session, leads_found=state['leads_found'] + results['total_leads'])

            if results['total_leads'] > 0:
                template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                if template:
                    kb_info = get_knowledge_base_info(session, get_active_project_id())
                    from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                    reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                    logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': res['Email']} for res in results['results']])
                    update_background_state(session, emails_sent=state['emails_sent'] + sent_count)

        update_background_state(session, last_run=datetime.utcnow())

def start_background_process():
    with safe_db_session() as session:
        update_background_state(session, is_running=True)

def pause_background_search():
    with safe_db_session() as session:
        update_background_state(session, is_running=False)

def resume_background_search():
    with safe_db_session() as session:
        update_background_state(session, is_running=True)

def stop_background_search():
    with safe_db_session() as session:
        update_background_state(session, is_running=False, current_term=None, job_progress=0)

def settings_page():
    st.title("Settings")
    
    with safe_db_session() as session:
        email_settings = fetch_email_settings(session)
        
        st.subheader("Email Settings")
        for setting in email_settings:
            with st.expander(f"{setting['name']} ({setting['email']})"):
                name = st.text_input("Name", value=setting['name'], key=f"name_{setting['id']}")
                email = st.text_input("Email", value=setting['email'], key=f"email_{setting['id']}")
                provider = st.selectbox("Provider", ["SMTP", "AWS SES"], key=f"provider_{setting['id']}")
                
                if provider == "SMTP":
                    smtp_server = st.text_input("SMTP Server", key=f"smtp_server_{setting['id']}")
                    smtp_port = st.number_input("SMTP Port", min_value=1, max_value=65535, key=f"smtp_port_{setting['id']}")
                    smtp_username = st.text_input("SMTP Username", key=f"smtp_username_{setting['id']}")
                    smtp_password = st.text_input("SMTP Password", type="password", key=f"smtp_password_{setting['id']}")
                else:
                    aws_access_key_id = st.text_input("AWS Access Key ID", key=f"aws_access_key_id_{setting['id']}")
                    aws_secret_access_key = st.text_input("AWS Secret Access Key", type="password", key=f"aws_secret_access_key_{setting['id']}")
                    aws_region = st.text_input("AWS Region", key=f"aws_region_{setting['id']}")
                
                if st.button("Update", key=f"update_{setting['id']}"):
                    update_email_setting(session, setting['id'], name, email, provider, smtp_server, smtp_port, smtp_username, smtp_password, aws_access_key_id, aws_secret_access_key, aws_region)
                    st.success("Email setting updated successfully!")
        
        if st.button("Add New Email Setting"):
            add_new_email_setting(session)
            st.success("New email setting added successfully!")
            st.rerun()

def update_email_setting(session, setting_id, name, email, provider, smtp_server=None, smtp_port=None, smtp_username=None, smtp_password=None, aws_access_key_id=None, aws_secret_access_key=None, aws_region=None):
    setting = session.query(EmailSettings).filter_by(id=setting_id).first()
    if setting:
        setting.name = name
        setting.email = email
        setting.provider = provider
        setting.smtp_server = smtp_server
        setting.smtp_port = smtp_port
        setting.smtp_username = smtp_username
        setting.smtp_password = smtp_password
        setting.aws_access_key_id = aws_access_key_id
        setting.aws_secret_access_key = aws_secret_access_key
        setting.aws_region = aws_region
        session.commit()

def add_new_email_setting(session):
    new_setting = EmailSettings(
        name="New Email Setting",
        email="example@example.com",
        provider="SMTP"
    )
    session.add(new_setting)
    session.commit()

def auto_refresh():
    # Placeholder implementation
    pass
def search_terms_page():
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Search Terms Dashboard</h1>", unsafe_allow_html=True)
    with safe_db_session() as session:
        search_terms_df = fetch_search_terms_with_lead_count(session)
        if not search_terms_df.empty:
            st.columns(3)[0].metric("Total Search Terms", len(search_terms_df))
            st.columns(3)[1].metric("Total Leads", search_terms_df['Lead Count'].sum())
            st.columns(3)[2].metric("Total Emails Sent", search_terms_df['Email Count'].sum())

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Search Term Groups", "Performance", "Add New Term", "AI Grouping", "Manage Groups"])

            with tab1:
                groups = session.query(SearchTermGroup).all()
                groups.append("Ungrouped")
                for group in groups:
                    with st.expander(group.name if isinstance(group, SearchTermGroup) else group, expanded=True):
                        group_id = group.id if isinstance(group, SearchTermGroup) else None
                        terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all() if group_id else session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
                        updated_terms = st_tags(
                            label="",
                            text="Add or remove terms",
                            value=[f"{term.id}: {term.term}" for term in terms],
                            suggestions=[term for term in search_terms_df['Term'] if term not in [f"{t.id}: {t.term}" for t in terms]],
                            key=f"group_{group_id}"
                        )
                        if st.button("Update", key=f"update_{group_id}"):
                            update_search_term_group(session, group_id, updated_terms)
                            st.success("Group updated successfully")
                            st.rerun()

            with tab2:
                col1, col2 = st.columns([3, 1])
                with col1:
                    chart_type = st.radio("Chart Type", ["Bar", "Pie"], horizontal=True)
                    fig = px.bar(search_terms_df.nlargest(10, 'Lead Count'), x='Term', y=['Lead Count', 'Email Count'], title='Top 10 Search Terms', labels={'value': 'Count', 'variable': 'Type'}, barmode='group') if chart_type == "Bar" else px.pie(search_terms_df, values='Lead Count', names='Term', title='Lead Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(search_terms_df.nlargest(5, 'Lead Count')[['Term', 'Lead Count', 'Email Count']], use_container_width=True)

            with tab3:
                col1, col2, col3 = st.columns([2,1,1])
                new_term = col1.text_input("New Search Term")
                campaign_id = get_active_campaign_id()
                group_for_new_term = col2.selectbox("Assign to Group", ["None"] + [f"{g.id}: {g.name}" for g in groups if isinstance(g, SearchTermGroup)], format_func=lambda x: x.split(":")[1] if ":" in x else x)
                if col3.button("Add Term", use_container_width=True) and new_term:
                    add_new_search_term(session, new_term, campaign_id, group_for_new_term)
                    st.success(f"Added: {new_term}")
                    st.rerun()

            with tab4:
                st.subheader("AI-Powered Search Term Grouping")
                ungrouped_terms = session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
                if ungrouped_terms:
                    st.write(f"Found {len(ungrouped_terms)} ungrouped search terms.")
                    if st.button("Group Ungrouped Terms with AI"):
                        with st.spinner("AI is grouping terms..."):
                            grouped_terms = ai_group_search_terms(session, ungrouped_terms)
                            update_search_term_groups(session, grouped_terms)
                            st.success("Search terms have been grouped successfully!")
                            st.rerun()
                else:
                    st.info("No ungrouped search terms found.")

            with tab5:
                st.subheader("Manage Search Term Groups")
                col1, col2 = st.columns(2)
                with col1:
                    new_group_name = st.text_input("New Group Name")
                    if st.button("Create New Group") and new_group_name:
                        create_search_term_group(session, new_group_name)
                        st.success(f"Created new group: {new_group_name}")
                        st.rerun()
                with col2:
                    group_to_delete = st.selectbox("Select Group to Delete", 
                                                   [f"{g.id}: {g.name}" for g in groups if isinstance(g, SearchTermGroup)],
                                                   format_func=lambda x: x.split(":")[1])
                    if st.button("Delete Group") and group_to_delete:
                        group_id = int(group_to_delete.split(":")[0])
                        delete_search_term_group(session, group_id)
                        st.success(f"Deleted group: {group_to_delete.split(':')[1]}")
                        st.rerun()

        else:
            st.info("No search terms available. Add some to your campaigns.")

def update_search_term_group(session, group_id, updated_terms):
    try:
        current_term_ids = set(int(term.split(":")[0]) for term in updated_terms)
        existing_terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all()

        for term in existing_terms:
            if term.id not in current_term_ids:
                term.group_id = None

        for term_str in updated_terms:
            term_id = int(term_str.split(":")[0])
            term = session.query(SearchTerm).get(term_id)
            if term:
                term.group_id = group_id

        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error in update_search_term_group: {str(e)}")

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try:
        new_search_term = SearchTerm(term=new_term, campaign_id=campaign_id, created_at=datetime.utcnow())
        if group_for_new_term != "None":
            new_search_term.group_id = int(group_for_new_term.split(":")[0])
        session.add(new_search_term)
        session.commit()
    except Exception as e:
        session.rollback()
        log_error(f"Error adding search term: {str(e)}")

def ai_group_search_terms(session, ungrouped_terms):
    existing_groups = session.query(SearchTermGroup).all()

    prompt = f"""
    Categorize these search terms into existing groups or suggest new ones:
    {', '.join([term.term for term in ungrouped_terms])}

    Existing groups: {', '.join([group.name for group in existing_groups])}

    Respond with a JSON object: {{group_name: [term1, term2, ...]}}
    """

    messages = [
        {"role": "system", "content": "You're an AI that categorizes search terms for lead generation. Be concise and efficient."},
        {"role": "user", "content": prompt}
    ]

    response = openai_chat_completion(messages, function_name="ai_group_search_terms")

    return response if isinstance(response, dict) else {}

def update_search_term_groups(session, grouped_terms):
    try:
        for group_name, terms in grouped_terms.items():
            group = session.query(SearchTermGroup).filter_by(name=group_name).first()
            if not group:
                group = SearchTermGroup(name=group_name)
                session.add(group)
                session.flush()

            for term in terms:
                search_term = session.query(SearchTerm).filter_by(term=term).first()
                if search_term:
                    search_term.group_id = group.id

        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Database error in update_search_term_groups: {str(e)}")
        raise

def create_search_term_group(session, group_name):
    try:
        new_group = SearchTermGroup(name=group_name)
        session.rollback()
        logging.error(f"Database error: {str(e)}")
        raise

def auto_perform_optimized_search(session, term, num_results):
    search_query = session.query(Lead).filter(Lead.email.ilike(f"%{term}%"))

    if session.bind.dialect.name == 'postgresql':
        search_query = search_query.with_hint(Lead, 'USE INDEX (ix_lead_email)')

    search_results = search_query.limit(num_results).all()

    return [
        {
            "id": lead.id,
            "email": lead.email,
            "first_name": lead.first_name,
            "last_name": lead.last_name,
            "company": lead.company,
            "position": lead.job_title,
            "country": lead.country,
            "industry": lead.industry
        }
        for lead in search_results
    ]

def main():
    st.title("Bulk Email Sending")
    with safe_db_session() as session:
        templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)
        if not templates or not email_settings:
            st.error("No email templates or settings available. Please set them up first.")
            return

        template_option = st.selectbox("Email Template", options=templates, format_func=lambda x: x.split(":")[1].strip())
        template_id = int(template_option.split(":")[0])
        template = session.query(EmailTemplate).filter_by(id=template_id).first()

        if not template:
            st.error("Selected template not found. Please choose a valid template.")
            return

        # ... (rest of the function remains unchanged)

def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
    if not email_settings:
        logging.error(f"No email settings found for {from_email}")
        return None, None

    tracking_id = str(uuid.uuid4())
    tracking_pixel_url = f"{TRACKING_URL}?{urlencode({'id': tracking_id, 'type': 'open'})}"
    wrapped_body = wrap_email_body(body)
    soup = BeautifulSoup(wrapped_body, 'html.parser')

    # Insert tracking pixel
    img_tag = soup.new_tag('img', src=tracking_pixel_url, width="1", height="1", style="display:none;")
    soup.body.append(img_tag)

    # Replace URLs for click tracking
    for a in soup.find_all('a', href=True):
        original_url = a['href']
        tracked_url = f"{TRACKING_URL}?{urlencode({'id': tracking_id, 'type': 'click', 'url': original_url})}"
        a['href'] = tracked_url

    tracked_body = str(soup)

    try:
        response = None
        if email_settings.provider == 'ses':
            if ses_client is None:
                aws_session = boto3.Session(
                    aws_access_key_id=email_settings.aws_access_key_id,
                    aws_secret_access_key=email_settings.aws_secret_access_key,
                    region_name=email_settings.aws_region
                )
                ses_client = aws_session.client('ses')

            response = ses_client.send_email(
                Source=from_email,
                Destination={'ToAddresses': [to_email]},
                Message={
                    'Subject': {'Data': subject, 'Charset': charset},
                    'Body': {'Html': {'Data': tracked_body, 'Charset': charset}}
                },
                ReplyToAddresses=[reply_to] if reply_to else []
            )
        elif email_settings.provider == 'smtp':
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            if reply_to:
                msg['Reply-To'] = reply_to
            msg.attach(MIMEText(tracked_body, 'html'))

            with smtplib.SMTP(email_settings.smtp_server, email_settings.smtp_port) as server:
                server.starttls()
                server.login(email_settings.smtp_username, email_settings.get_password())
                server.send_message(msg)
            response = {'MessageId': f'smtp-{uuid.uuid4()}'}
        else:
            logging.error(f"Unknown email provider: {email_settings.provider}")

        return response, tracking_id
    except (ClientError, smtplib.SMTPException) as e:
        logging.error(f"Error sending email: {str(e)}")
        return None, tracking_id

        session.query(LeadSource).filter(LeadSource.lead_id == lead_id).delete()
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            session.delete(lead)
        session.commit()
        return True
    except SQLAlchemyError as e:
        logging.error(f"Error deleting lead {lead_id} and its sources: {str(e)}")
        session.rollback()
        return False

def background_manual_search(session_factory):
    with safe_db_session() as session:
        state = get_background_state()
        if not state['is_running']:
            return

        search_terms = session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()
        for term in search_terms:
            update_background_state(session, current_term=term.term)
            results = asyncio.run(manual_search(session, [term.term], 10, ignore_previously_fetched=True))
            update_background_state(session, leads_found=state['leads_found'] + results['total_leads'])

            if results['total_leads'] > 0:
                template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                if template:
                    kb_info = get_knowledge_base_info(session, get_active_project_id())
                    from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                    reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                    logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': res['Email']} for res in results['results']])
                    update_background_state(session, emails_sent=state['emails_sent'] + sent_count)

        update_background_state(session, last_run=datetime.utcnow())

async def ai_automation_loop(session, log_container, leads_container):
    automation_logs, total_search_terms, total_emails_sent = [], 0, 0
    max_iterations = 100
    iteration = 0
    while st.session_state.get('automation_status', False) and iteration < max_iterations:
        try:
            log_container.info("Starting automation cycle")
            kb_info = get_knowledge_base_info(session, get_active_project_id())
            if not kb_info:
                log_container.warning("Knowledge Base not found. Skipping cycle.")
                await asyncio.sleep(3600)
                continue
            base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
            optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
            st.subheader("Optimized Search Terms")
            st.write(", ".join(optimized_terms))

            total_search_terms = len(optimized_terms)
            progress_bar = st.progress(0)
            for idx, term in enumerate(optimized_terms):
                results = await manual_search(session, [term], 10, ignore_previously_fetched=True)
                new_leads = []
                for res in results['results']:
                    lead = save_lead(session, res['Email'], url=res['URL'])
                    if lead:
                        new_leads.append((lead.id, lead.email))
                if new_leads:
                    template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                    if template:
                        from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                        reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                        logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': email} for _, email in new_leads])
                        automation_logs.extend(logs)
                        total_emails_sent += sent_count
                leads_container.text_area("New Leads Found", "\n".join([email for _, email in new_leads]), height=200)
                progress_bar.progress((idx + 1) / len(optimized_terms))
            st.success(f"Automation cycle completed. Total search terms: {total_search_terms}, Total emails sent: {total_emails_sent}")
            iteration += 1
            await asyncio.sleep(3600)
        except Exception as e:
            log_container.error(f"Critical error in automation cycle: {str(e)}")
            await asyncio.sleep(300)
    
    if iteration >= max_iterations:
        log_container.warning("Maximum iterations reached. Stopping automation.")
    st.session_state.automation_status = False
    st.session_state.automation_logs = automation_logs
    st.session_state.total_leads_found = total_search_terms
    st.session_state.total_emails_sent = total_emails_sent

def projects_campaigns_page():
    with safe_db_session() as session:
        st.header("Projects and Campaigns")
        
        # Add New Project
        with st.form("add_project_form"):
            project_name = st.text_input("Project Name")
            if st.form_submit_button("Add Project"):
                if project_name.strip() and re.match(r'^[A-Za-z0-9 _-]{3,50}$', project_name):
                    try:
                        session.add(Project(project_name=project_name, created_at=datetime.utcnow()))
                        session.commit()
                        st.success(f"Project '{project_name}' added successfully.")
                    except SQLAlchemyError as e:
                        st.error(f"Error adding project: {str(e)}")
                else:
                    st.warning("Project name must be 3-50 characters long and contain only letters, numbers, spaces, underscores, or hyphens.")
        
        # Existing Projects and Campaigns
        st.subheader("Existing Projects and Campaigns")
        projects = session.query(Project).all()
        for project in projects:
            with st.expander(f"{project.project_name} (ID: {project.id})"):
                st.write(f"Created At: {project.created_at}")
                st.write(f"Total Campaigns: {len(project.campaigns)}")
                st.write(f"Knowledge Base: {'Configured' if project.knowledge_base else 'Not Configured'}")
                
                # Display Campaigns
                campaigns = session.query(Campaign).filter_by(project_id=project.id).all()
                for campaign in campaigns:
                    with st.expander(f"Campaign: {campaign.campaign_name} (ID: {campaign.id})"):
                        st.write(f"Type: {campaign.campaign_type}")
                        st.write(f"Created At: {campaign.created_at}")
                        st.write(f"Emails Sent: {campaign.emails_sent}")
                        st.write(f"Leads Found: {campaign.leads_found}")
                        if st.button(f"Delete Campaign {campaign.id}", key=f"delete_campaign_{campaign.id}"):
                            try:
                                session.delete(campaign)
                                session.commit()
                                st.success(f"Campaign {campaign.id} deleted successfully.")
                                st.experimental_rerun()
                            except SQLAlchemyError as e:
                                st.error(f"Error deleting campaign: {str(e)}")
                
                # Add New Campaign
                with st.form(f"add_campaign_form_{project.id}"):
                    campaign_name = st.text_input("Campaign Name", key=f"campaign_name_{project.id}")
                    campaign_type = st.selectbox("Campaign Type", options=["Email", "SMS", "Social Media"], key=f"campaign_type_{project.id}")
                    if st.form_submit_button("Add Campaign", key=f"add_campaign_{project.id}"):
                        if campaign_name.strip() and re.match(r'^[A-Za-z0-9 _-]{3,50}$', campaign_name):
                            try:
                                session.add(Campaign(
                                    campaign_name=campaign_name,
                                    campaign_type=campaign_type,
                                    project_id=project.id,
                                    created_at=datetime.utcnow()
                                ))
                                session.commit()
                                st.success(f"Campaign '{campaign_name}' added successfully.")
                            except SQLAlchemyError as e:
                                st.error(f"Error adding campaign: {str(e)}")
                        else:
                            st.warning("Campaign name must be 3-50 characters long and contain only letters, numbers, spaces, underscores, or hyphens.")

def update_search_terms(session, classified_terms):
    for group, terms in classified_terms.items():
        for term in terms:
            existing_term = session.query(SearchTerm).filter_by(term=term, project_id=get_active_project_id()).first()
            if existing_term:
                existing_term.group = group
            else:
                session.add(SearchTerm(term=term, group=group, project_id=get_active_project_id()))
    session.commit()

def display_results_and_logs(container, items, title, item_type):
    if not items:
        container.info(f"No {item_type} to display yet.")
        return

    container.markdown(
        f"""
        <style>
        .results-container {{
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: 1rem;
        }}
        .result-entry {{
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            border-radius: 0.25rem;
            background-color: rgba(28, 131, 225, 0.1);
        }}
        </style>
        <h4>{title}</h4>
        <div class="results-container">
        {"".join(f'<div class="result-entry">{item}</div>' for item in items[-20:])}
        </div>
        """,
        unsafe_allow_html=True
    )

def manual_search_page():
    with safe_db_session() as session:
        st.title("Manual Search")

        # Input Section
        search_terms = st_tags(
            label='Enter search terms:',
            text='Press enter to add more',
            value=[],
            key='search_terms_input'
        )
        num_results = st.slider("Results per term", 1, 500, 10)
        
        # Additional Options
        col1, col2 = st.columns(2)
        with col1:
            ignore_previously_fetched = st.checkbox("Ignore fetched domains", value=True)
            shuffle_keywords = st.checkbox("Shuffle Keywords", value=False)
        with col2:
            optimize_english = st.checkbox("Optimize (English)", value=False)
            optimize_spanish = st.checkbox("Optimize (Spanish)", value=False)

        # Run Manual Search Button
        if st.button("Run Manual Search"):
            if not search_terms:
                st.warning("Please enter at least one search term.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.empty()

                try:
                    results = manual_search(
                        session, search_terms, num_results,
                        ignore_previously_fetched=ignore_previously_fetched,
                        optimize_english=optimize_english,
                        optimize_spanish=optimize_spanish,
                        shuffle_keywords_option=shuffle_keywords,
                        enable_email_sending=user_settings['enable_email_sending'],
                        from_email=from_email if enable_email_sending else None,
                        email_template=email_template.split(":")[0] if enable_email_sending else None,
                        reply_to=reply_to if enable_email_sending else None,
                        log_container=log_container,
                        progress_bar=progress_bar,
                        status_text=status_text
                    )
                    st.write(f"Total leads found: {results['total_leads']}")
                    st.dataframe(results['results'])
                except Exception as e:
                    st.error(f"An error occurred during manual search: {str(e)}")
                    
def manual_search(session, terms, num_results, **kwargs):
    progress_bar = kwargs.get('progress_bar')
    status_text = kwargs.get('status_text')
    log_container = kwargs.get('log_container')
    
    results = []
    total_leads = 0
    
    for i, term in enumerate(terms):
        if progress_bar:
            progress_bar.progress((i + 1) / len(terms))
        if status_text:
            status_text.text(f"Searching term {i+1}/{len(terms)}: {term}")
        
        try:
            search_term_id = add_or_get_search_term(session, term, get_active_campaign_id())
            optimized_term = shuffle_keywords(term) if kwargs.get('shuffle_keywords_option') else term
            optimized_term = optimize_search_term(
                optimized_term,
                'english' if kwargs.get('optimize_english') else 'spanish'
            ) if kwargs.get('optimize_english') or kwargs.get('optimize_spanish') else optimized_term
            
            log_container.text(f"Searching for '{term}' (Optimized: '{optimized_term}')")
            
            for url in optimized_search(optimized_term, num_results):
                domain = get_domain_from_url(url)
                if kwargs.get('ignore_previously_fetched') and domain in domains_processed:
                    log_container.warning(f"Skipping previously fetched domain: {domain}")
                    continue
                
                try:
                    response = safe_request(url)
                    if response is None:
                        continue
                    emails = extract_emails_from_html(response.text, domain)
                    log_container.success(f"Found {len(emails)} email(s) on {url}")
                    
                    for email_info in emails:
                        email = email_info['email']
                        if (kwargs.get('one_email_per_url') and email in emails_processed) or \
                           (kwargs.get('one_email_per_domain') and email.split('@')[1] in domains_processed):
                            continue
                        
                        emails_processed.add(email)
                        domains_processed.add(email.split('@')[1])
                        
                        lead = save_lead(session, email, email_info)
                        if lead:
                            total_leads += 1
                            results.append({
                                'Email': email,
                                'URL': domain,
                                'Lead Source': term,
                                'Title': email_info['page_title'],
                                'Description': email_info['meta_description'],
                                'Tags': email_info['tags'],
                                'Name': email_info['name'],
                                'Company': email_info['company'],
                                'Job Title': email_info['job_title'],
                                'Search Term ID': search_term_id
                            })
                            log_container.success(f"Saved lead: {email}")
                            
                            if kwargs.get('enable_email_sending'):
                                if not from_email or not email_template:
                                    log_container.error("Email sending is enabled but from_email or email_template is not provided.")
                                    continue

                                template = session.query(EmailTemplate).filter_by(id=int(email_template)).first()
                                if not template:
                                    log_container.error("Email template not found.")
                                    continue

                                wrapped_content = wrap_email_body(template.body_content)
                                try:
                                    response, tracking_id = rate_limited_send_email(session, from_email, email, template.subject, wrapped_content, reply_to=reply_to)
                                    status = 'sent' if response else 'failed'
                                    results.append({'Email': email, 'Status': status})
                                    if response:
                                        log_container.info(f"Sent email to: {email}")
                                    else:
                                        log_container.error(f"Failed to send email to: {email}")
                                except Exception as e:
                                    log_container.error(f"Error sending email to {email}: {str(e)}")
                except requests.RequestException as e:
                    log_container.error(f"Error fetching URL {url}: {str(e)}")
                    continue
        except Exception as e:
            log_container.error(f"Error processing term '{term}': {str(e)}")
    
    return {"total_leads": total_leads, "results": results}

def bulk_send_page():
    with safe_db_session() as session:
        st.title("Bulk Email Sending")
        
        # Select Email Template
        email_templates = fetch_email_templates(session)
        email_template = st.selectbox("Select Email Template", options=email_templates, format_func=lambda x: x.split(":")[1].strip())
        if email_template:
            template_id = int(email_template.split(":")[0])
        else:
            st.error("Please select an email template.")
            return
        
        # Select Leads
        send_option = st.radio("Send to:", ["All Leads", "Specific Email", "Leads from Chosen Search Terms"], index=0)
        specific_email = None
        selected_terms = None
        if send_option == "Specific Email":
            specific_email = st.text_input("Enter Email Address")
            if specific_email and not is_valid_email(specific_email):
                st.error("Please enter a valid email address.")
        elif send_option == "Leads from Chosen Search Terms":
            search_terms = fetch_search_terms(session)
            selected_terms = st.multiselect("Select Search Terms", options=search_terms)
        
        # Email Options
        email_settings = fetch_email_settings(session)
        if not email_settings:
            st.error("No email settings available. Please configure them first.")
            return
        from_email_option = st.selectbox("From Email", options=email_settings, format_func=lambda x: f"{x['name']} ({x['email']})")
        if from_email_option:
            from_email = from_email_option['email']
            reply_to = st.text_input("Reply-To", value=from_email)
        else:
            st.error("Please select a valid 'From Email'.")
            return
        
        # Send Emails Button
        if st.button("Send Emails", type="primary"):
            leads = fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted=True)
            total_leads = len(leads)
            
            if total_leads == 0:
                st.warning("No leads found matching the selected criteria.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.empty()
                results = []
                
                try:
                    logs, sent_count = bulk_send_emails(
                        session, template_id, from_email, reply_to, leads,
                        progress_bar=progress_bar,
                        status_text=status_text,
                        results=results,
                        log_container=log_container
                    )
                    st.success(f"Emails sent successfully to {sent_count} leads.")
                    st.subheader("Sending Results")
                    st.dataframe(pd.DataFrame(results))
                    success_rate = (pd.DataFrame(results)['Status'] == 'sent').mean() * 100
                    st.metric("Email Sending Success Rate", f"{success_rate:.2f}%")
                except Exception as e:
                    st.error(f"An error occurred during bulk email sending: {str(e)}")
                    
def bulk_send_emails(session, template_id, from_email, reply_to, leads, **kwargs):
    progress_bar = kwargs.get('progress_bar')
    status_text = kwargs.get('status_text')
    log_container = kwargs.get('log_container')
    results = kwargs.get('results', [])
    
    sent_count = 0
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
        log_container.error(f"Error in bulk email sending: {str(e)}")
    finally:
        return {"logs": results, "sent_count": sent_count}

def view_campaign_logs():
    with safe_db_session() as session:
        st.header("Email Logs")
        logs = fetch_all_email_logs(session)
        if logs.empty:
            st.info("No email logs found.")
        else:
            st.write(f"Total emails sent: {len(logs)}")
            st.write(f"Success rate: {(logs['Status'] == 'sent').mean():.2%}")

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=logs['Sent At'].min().date())
            with col2:
                end_date = st.date_input("End Date", value=logs['Sent At'].max().date())

            filtered_logs = logs[(logs['Sent At'].dt.date >= start_date) & (logs['Sent At'].dt.date <= end_date)]

            search_term = st.text_input("Search by email or subject")
            if search_term:
                filtered_logs = filtered_logs[
                    filtered_logs['Email'].str.contains(search_term, case=False, na=False) | 
                    filtered_logs['Subject'].str.contains(search_term, case=False, na=False)
                ]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Emails Sent", len(filtered_logs))
            with col2:
                st.metric("Unique Recipients", filtered_logs['Email'].nunique())
            with col3:
                st.metric("Success Rate", f"{(filtered_logs['Status'] == 'sent').mean():.2%}")

            daily_counts = filtered_logs.resample('D', on='Sent At')['Email'].count()
            st.bar_chart(daily_counts)

            st.subheader("Detailed Email Logs")
            for _, log in filtered_logs.iterrows():
                with st.expander(f"{log['Sent At'].strftime('%Y-%m-%d %H:%M:%S')} - {log['Email']} - {log['Status']}"):
                    st.write(f"**Subject:** {log['Subject']}")
                    st.write(f"**Content Preview:** {log['Content'][:100]}...")
                    if st.button("View Full Email", key=f"view_email_{log['ID']}"):
                        st.components.v1.html(wrap_email_body(log['Content']), height=400, scrolling=True)
                    if log['Status'] != 'sent':
                        st.error(f"Status: {log['Status']}")

            logs_per_page = 20
            total_pages = (len(filtered_logs) - 1) // logs_per_page + 1
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            start_idx = (page - 1) * logs_per_page
            end_idx = start_idx + logs_per_page

            st.table(filtered_logs.iloc[start_idx:end_idx][['Sent At', 'Email', 'Subject', 'Status']])

def fetch_all_email_logs(session):
    try:
        query = session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(
            joinedload(EmailCampaign.lead),
            joinedload(EmailCampaign.template)
        ).order_by(EmailCampaign.sent_at.desc())
        
        data = [{
            'ID': ec.id,
            'Sent At': ec.sent_at,
            'Email': ec.lead.email,
            'Template': ec.template.template_name,
            'Subject': ec.customized_subject or "No subject",
            'Content': ec.customized_content or "No content",
            'Status': ec.status,
            'Message ID': ec.message_id or "No message ID",
            'Campaign ID': ec.campaign_id,
            'Lead Name': f"{ec.lead.first_name or ''} {ec.lead.last_name or ''}".strip() or "Unknown",
            'Lead Company': ec.lead.company or "Unknown"
        } for ec in query.all()]
        
        return pd.DataFrame(data)
    except SQLAlchemyError as e:
        st.error(f"Database error while fetching email logs: {str(e)}")
        return pd.DataFrame()

def wrap_email_body(content):
    return f"""
    <html>
        <body>
            {content}
        </body>
    </html>
    """

def enqueue_job(session, job_type, job_params):
    state = get_background_state()
    if not state['is_running']:
        update_background_state(session, 
                                is_running=True, 
                                job_type=job_type, 
                                job_params=job_params, 
                                job_progress=0)
        return True
    return False

def auto_perform_optimized_search(session, term, num_results):
    search_query = session.query(Lead).filter(Lead.email.ilike(f"%{term}%"))

    if session.bind.dialect.name == 'postgresql':
        search_query = search_query.with_hint(Lead, 'USE INDEX (ix_lead_email)')

    search_results = search_query.limit(num_results).all()

    return [
        {
            "id": lead.id,
            "email": lead.email,
            "first_name": lead.first_name,
            "last_name": lead.last_name,
            "company": lead.company,
            "position": lead.job_title,
            "country": lead.country,
            "industry": lead.industry
        }
        for lead in search_results
    ]

def main():
    if 'automation_status' not in st.session_state:
        st.session_state.automation_status = False
    if 'automation_logs' not in st.session_state:
        st.session_state.automation_logs = []
    if 'total_leads_found' not in st.session_state:
        st.session_state.total_leads_found = 0
    if 'total_emails_sent' not in st.session_state:
        st.session_state.total_emails_sent = 0
    
    # Initialize the scheduler once in the main function
    scheduler = BackgroundScheduler()
    scheduler.start()

    scheduler_lock = threading.Lock()

    def add_job_safely(func, *args, **kwargs):
        with scheduler_lock:
            scheduler.add_job(func, *args, **kwargs)

    # Add background search job
    add_job_safely(
        func=background_manual_search,
        args=[SessionLocal],
        trigger=IntervalTrigger(minutes=60),
        id='background_search',
        name='Background Search',
        replace_existing=True
    )

    # ... (rest of the main function)

    auto_refresh()

if __name__ == "__main__":
    main()

def send_email_ses(session, from_email, to_email, subject, body, charset='UTF-8', reply_to=None, ses_client=None):
    email_settings = session.query(EmailSettings).filter_by(email=from_email).first()
    if not email_settings:
        logging.error(f"No email settings found for {from_email}")
        return None, None

    tracking_id = str(uuid.uuid4())
    tracking_pixel_url = f"{TRACKING_URL}?{urlencode({'id': tracking_id, 'type': 'open'})}"
    wrapped_body = wrap_email_body(body)
    soup = BeautifulSoup(wrapped_body, 'html.parser')

    # Insert tracking pixel
    img_tag = soup.new_tag('img', src=tracking_pixel_url, width="1", height="1", style="display:none;")
    soup.body.append(img_tag)

    # Replace URLs for click tracking
    for a in soup.find_all('a', href=True):
        original_url = a['href']
        tracked_url = f"{TRACKING_URL}?{urlencode({'id': tracking_id, 'type': 'click', 'url': original_url})}"
        a['href'] = tracked_url

    tracked_body = str(soup)

    try:
        response = None
        if email_settings.provider == 'ses':
            if ses_client is None:
                aws_session = boto3.Session(
                    aws_access_key_id=email_settings.aws_access_key_id,
                    aws_secret_access_key=email_settings.aws_secret_access_key,
                    region_name=email_settings.aws_region
                )
                ses_client = aws_session.client('ses')

            response = ses_client.send_email(
                Source=from_email,
                Destination={'ToAddresses': [to_email]},
                Message={
                    'Subject': {'Data': subject, 'Charset': charset},
                    'Body': {'Html': {'Data': tracked_body, 'Charset': charset}}
                },
                ReplyToAddresses=[reply_to] if reply_to else []
            )
        elif email_settings.provider == 'smtp':
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            if reply_to:
                msg['Reply-To'] = reply_to
            msg.attach(MIMEText(tracked_body, 'html'))

            with smtplib.SMTP(email_settings.smtp_server, email_settings.smtp_port) as server:
                server.starttls()
                server.login(email_settings.smtp_username, email_settings.get_password())
                server.send_message(msg)
            response = {'MessageId': f'smtp-{uuid.uuid4()}'}
        else:
            logging.error(f"Unknown email provider: {email_settings.provider}")

        return response, tracking_id
    except (ClientError, smtplib.SMTPException) as e:
        logging.error(f"Error sending email: {str(e)}")
        return None, tracking_id

def save_email_campaign(session, lead_email, template_id, status, sent_at, subject, message_id, email_body):
    try:
        lead = session.query(Lead).filter_by(email=lead_email).first()
        if not lead:
            logging.error(f"Lead with email {lead_email} not found.")
            return

        new_campaign = EmailCampaign(
            lead_id=lead.id,
            template_id=template_id,
            status=status,
            sent_at=sent_at,
            customized_subject=subject or "No subject",
            message_id=message_id or f"unknown-{uuid.uuid4()}",
            customized_content=email_body or "No content",
            campaign_id=get_active_campaign_id(),
            tracking_id=str(uuid.uuid4())
        )
        session.add(new_campaign)
        session.commit()
    except Exception as e:
        logging.error(f"Error saving email campaign: {str(e)}")
        session.rollback()

def update_log(log_container, message, level='info'):
    icon = {'info': '🔵', 'success': '🟢', 'warning': '🟠', 'error': '🔴', 'email_sent': '🟣'}.get(level, '⚪')
    log_entry = f"{icon} {message}"

    # Log to file or console without HTML
    logging.log(getattr(logging, level.upper(), logging.INFO), message)

    if 'log_entries' not in st.session_state:
        st.session_state.log_entries = []

    st.session_state.log_entries.append(log_entry)

    if log_container is not None:
        log_container.markdown("\n".join(st.session_state.log_entries))

def optimize_search_term(search_term, language):
    if language == 'english':
        return f'"{search_term}" email OR contact OR "get in touch" site:.com'
    elif language == 'spanish':
        return f'"{search_term}" correo OR contacto OR "ponte en contacto" site:.es'
    return search_term

def shuffle_keywords(term):
    words = term.split()
    random.shuffle(words)
    return ' '.join(words)

def get_domain_from_url(url):
    return urlparse(url).netloc

def is_valid_email(email):
    try:
        validate_email(email)
        return True
    except EmailNotValidError:
        return False

def extract_emails_from_html(html_content, domain):
    soup = BeautifulSoup(html_content, 'html.parser')
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, html_content)

    page_title = soup.title.string if soup.title else "No title found"
    meta_description = soup.find('meta', attrs={'name': 'description'})
    meta_description = meta_description['content'] if meta_description else "No description found"

    results = []
    for email in emails:
        name, company, job_title = extract_info_from_page(soup)
        results.append({
            'email': email,
            'name': name,
            'company': company,
            'job_title': job_title,
            'page_title': page_title,
            'meta_description': meta_description,
            'tags': [tag.name for tag in soup.find_all()],
            'domain': domain
        })

    return results

def extract_info_from_page(soup):
    name = soup.find('meta', {'name': 'author'})
    name = name['content'] if name else ''

    company = soup.find('meta', {'property': 'og:site_name'})
    company = company['content'] if company else ''

    job_title = soup.find('meta', {'name': 'job_title'})
    job_title = job_title['content'] if job_title else ''

    return name, company, job_title
@functools.lru_cache(maxsize=100)
def optimized_search(term, num_results):
    # Implement more efficient search algorithm here
    # This is a placeholder for demonstration
    return google_search(term, num_results)

async def async_fetch(url, session):
    try:
        async with session.get(url, timeout=10) as response:
            return await response.text()
    except Exception as e:
        logging.error(f"Error fetching {url}: {str(e)}")
        return None

async def manual_search(session, terms, num_results, **kwargs):
    results, total_leads = [], 0
    async with aiohttp.ClientSession() as client:
        for term in terms:
            search_results = await asyncio.gather(*[async_fetch(url, client) for url in optimized_search(term, num_results)])
            for html_content in search_results:
                if html_content:
                    emails = extract_emails_from_html(html_content, get_domain_from_url(url))
                    # Process emails and update results
                    # ... (rest of the code)
    return {"total_leads": total_leads, "results": results}

def delete_lead(session, lead_id):
    try:
        session.query(LeadSource).filter(LeadSource.lead_id == lead_id).delete()
        lead = session.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            session.delete(lead)
        session.commit()
        return True
    except SQLAlchemyError as e:
        logging.error(f"Error deleting lead {lead_id} and its sources: {str(e)}")
        session.rollback()
        return False

def background_manual_search(session_factory):
    with safe_db_session() as session:
        state = get_background_state()
        if not state['is_running']:
            return

        search_terms = session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()
        for term in search_terms:
            update_background_state(session, current_term=term.term)
            results = asyncio.run(manual_search(session, [term.term], 10, ignore_previously_fetched=True))
            update_background_state(session, leads_found=state['leads_found'] + results['total_leads'])

            if results['total_leads'] > 0:
                template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                if template:
                    kb_info = get_knowledge_base_info(session, get_active_project_id())
                    from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                    reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                    logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': res['Email']} for res in results['results']])
                    update_background_state(session, emails_sent=state['emails_sent'] + sent_count)

        update_background_state(session, last_run=datetime.utcnow())

def start_background_process():
    with safe_db_session() as session:
        update_background_state(session, is_running=True)

def pause_background_search():
    with safe_db_session() as session:
        update_background_state(session, is_running=False)

def resume_background_search():
    with safe_db_session() as session:
        update_background_state(session, is_running=True)

def stop_background_search():
    with safe_db_session() as session:
        update_background_state(session, is_running=False, current_term=None, job_progress=0)

def settings_page():
    st.title("Settings")
    
    with safe_db_session() as session:
        email_settings = fetch_email_settings(session)
        
        st.subheader("Email Settings")
        for setting in email_settings:
            with st.expander(f"{setting['name']} ({setting['email']})"):
                name = st.text_input("Name", value=setting['name'], key=f"name_{setting['id']}")
                email = st.text_input("Email", value=setting['email'], key=f"email_{setting['id']}")
                provider = st.selectbox("Provider", ["SMTP", "AWS SES"], key=f"provider_{setting['id']}")
                
                if provider == "SMTP":
                    smtp_server = st.text_input("SMTP Server", key=f"smtp_server_{setting['id']}")
                    smtp_port = st.number_input("SMTP Port", min_value=1, max_value=65535, key=f"smtp_port_{setting['id']}")
                    smtp_username = st.text_input("SMTP Username", key=f"smtp_username_{setting['id']}")
                    smtp_password = st.text_input("SMTP Password", type="password", key=f"smtp_password_{setting['id']}")
                else:
                    aws_access_key_id = st.text_input("AWS Access Key ID", key=f"aws_access_key_id_{setting['id']}")
                    aws_secret_access_key = st.text_input("AWS Secret Access Key", type="password", key=f"aws_secret_access_key_{setting['id']}")
                    aws_region = st.text_input("AWS Region", key=f"aws_region_{setting['id']}")
                
                if st.button("Update", key=f"update_{setting['id']}"):
                    update_email_setting(session, setting['id'], name, email, provider, smtp_server, smtp_port, smtp_username, smtp_password, aws_access_key_id, aws_secret_access_key, aws_region)
                    st.success("Email setting updated successfully!")
        
        if st.button("Add New Email Setting"):
            add_new_email_setting(session)
            st.success("New email setting added successfully!")
            st.rerun()

def update_email_setting(session, setting_id, name, email, provider, smtp_server=None, smtp_port=None, smtp_username=None, smtp_password=None, aws_access_key_id=None, aws_secret_access_key=None, aws_region=None):
    setting = session.query(EmailSettings).filter_by(id=setting_id).first()
    if setting:
        setting.name = name
        setting.email = email
        setting.provider = provider
        setting.smtp_server = smtp_server
        setting.smtp_port = smtp_port
        setting.smtp_username = smtp_username
        setting.smtp_password = smtp_password
        setting.aws_access_key_id = aws_access_key_id
        setting.aws_secret_access_key = aws_secret_access_key
        setting.aws_region = aws_region
        session.commit()

def add_new_email_setting(session):
    new_setting = EmailSettings(
        name="New Email Setting",
        email="example@example.com",
        provider="SMTP"
    )
    session.add(new_setting)
    session.commit()

def auto_refresh():
    # Placeholder implementation
    pass
def search_terms_page():
    st.markdown("<h1 style='text-align: center; color: #1E88E5;'>Search Terms Dashboard</h1>", unsafe_allow_html=True)
    with safe_db_session() as session:
        search_terms_df = fetch_search_terms_with_lead_count(session)
        if not search_terms_df.empty:
            st.columns(3)[0].metric("Total Search Terms", len(search_terms_df))
            st.columns(3)[1].metric("Total Leads", search_terms_df['Lead Count'].sum())
            st.columns(3)[2].metric("Total Emails Sent", search_terms_df['Email Count'].sum())

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Search Term Groups", "Performance", "Add New Term", "AI Grouping", "Manage Groups"])

            with tab1:
                groups = session.query(SearchTermGroup).all()
                groups.append("Ungrouped")
                for group in groups:
                    with st.expander(group.name if isinstance(group, SearchTermGroup) else group, expanded=True):
                        group_id = group.id if isinstance(group, SearchTermGroup) else None
                        terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all() if group_id else session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
                        updated_terms = st_tags(
                            label="",
                            text="Add or remove terms",
                            value=[f"{term.id}: {term.term}" for term in terms],
                            suggestions=[term for term in search_terms_df['Term'] if term not in [f"{t.id}: {t.term}" for t in terms]],
                            key=f"group_{group_id}"
                        )
                        if st.button("Update", key=f"update_{group_id}"):
                            update_search_term_group(session, group_id, updated_terms)
                            st.success("Group updated successfully")
                            st.rerun()

            with tab2:
                col1, col2 = st.columns([3, 1])
                with col1:
                    chart_type = st.radio("Chart Type", ["Bar", "Pie"], horizontal=True)
                    fig = px.bar(search_terms_df.nlargest(10, 'Lead Count'), x='Term', y=['Lead Count', 'Email Count'], title='Top 10 Search Terms', labels={'value': 'Count', 'variable': 'Type'}, barmode='group') if chart_type == "Bar" else px.pie(search_terms_df, values='Lead Count', names='Term', title='Lead Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.dataframe(search_terms_df.nlargest(5, 'Lead Count')[['Term', 'Lead Count', 'Email Count']], use_container_width=True)

            with tab3:
                col1, col2, col3 = st.columns([2,1,1])
                new_term = col1.text_input("New Search Term")
                campaign_id = get_active_campaign_id()
                group_for_new_term = col2.selectbox("Assign to Group", ["None"] + [f"{g.id}: {g.name}" for g in groups if isinstance(g, SearchTermGroup)], format_func=lambda x: x.split(":")[1] if ":" in x else x)
                if col3.button("Add Term", use_container_width=True) and new_term:
                    add_new_search_term(session, new_term, campaign_id, group_for_new_term)
                    st.success(f"Added: {new_term}")
                    st.rerun()

            with tab4:
                st.subheader("AI-Powered Search Term Grouping")
                ungrouped_terms = session.query(SearchTerm).filter(SearchTerm.group_id == None).all()
                if ungrouped_terms:
                    st.write(f"Found {len(ungrouped_terms)} ungrouped search terms.")
                    if st.button("Group Ungrouped Terms with AI"):
                        with st.spinner("AI is grouping terms..."):
                            grouped_terms = ai_group_search_terms(session, ungrouped_terms)
                            update_search_term_groups(session, grouped_terms)
                            st.success("Search terms have been grouped successfully!")
                            st.rerun()
                else:
                    st.info("No ungrouped search terms found.")

            with tab5:
                st.subheader("Manage Search Term Groups")
                col1, col2 = st.columns(2)
                with col1:
                    new_group_name = st.text_input("New Group Name")
                    if st.button("Create New Group") and new_group_name:
                        create_search_term_group(session, new_group_name)
                        st.success(f"Created new group: {new_group_name}")
                        st.rerun()
                with col2:
                    group_to_delete = st.selectbox("Select Group to Delete", 
                                                   [f"{g.id}: {g.name}" for g in groups if isinstance(g, SearchTermGroup)],
                                                   format_func=lambda x: x.split(":")[1])
                    if st.button("Delete Group") and group_to_delete:
                        group_id = int(group_to_delete.split(":")[0])
                        delete_search_term_group(session, group_id)
                        st.success(f"Deleted group: {group_to_delete.split(':')[1]}")
                        st.rerun()

        else:
            st.info("No search terms available. Add some to your campaigns.")

def update_search_term_group(session, group_id, updated_terms):
    try:
        current_term_ids = set(int(term.split(":")[0]) for term in updated_terms)
        existing_terms = session.query(SearchTerm).filter(SearchTerm.group_id == group_id).all()

        for term in existing_terms:
            if term.id not in current_term_ids:
                term.group_id = None

        for term_str in updated_terms:
            term_id = int(term_str.split(":")[0])
            term = session.query(SearchTerm).get(term_id)
            if term:
                term.group_id = group_id

        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Error in update_search_term_group: {str(e)}")

def add_new_search_term(session, new_term, campaign_id, group_for_new_term):
    try:
        new_search_term = SearchTerm(term=new_term, campaign_id=campaign_id, created_at=datetime.utcnow())
        if group_for_new_term != "None":
            new_search_term.group_id = int(group_for_new_term.split(":")[0])
        session.add(new_search_term)
        session.commit()
    except Exception as e:
        session.rollback()
        log_error(f"Error adding search term: {str(e)}")

def ai_group_search_terms(session, ungrouped_terms):
    existing_groups = session.query(SearchTermGroup).all()

    prompt = f"""
    Categorize these search terms into existing groups or suggest new ones:
    {', '.join([term.term for term in ungrouped_terms])}

    Existing groups: {', '.join([group.name for group in existing_groups])}

    Respond with a JSON object: {{group_name: [term1, term2, ...]}}
    """

    messages = [
        {"role": "system", "content": "You're an AI that categorizes search terms for lead generation. Be concise and efficient."},
        {"role": "user", "content": prompt}
    ]

    response = openai_chat_completion(messages, function_name="ai_group_search_terms")

    return response if isinstance(response, dict) else {}

def update_search_term_groups(session, grouped_terms):
    for group_name, terms in grouped_terms.items():
        group = session.query(SearchTermGroup).filter_by(name=group_name).first()
        if not group:
            group = SearchTermGroup(name=group_name)
            session.add(group)
            session.flush()

        for term in terms:
            search_term = session.query(SearchTerm).filter_by(term=term).first()
            if search_term:
                search_term.group_id = group.id

    session.commit()

def create_search_term_group(session, group_name):
    try:
        new_group = SearchTermGroup(name=group_name)
        session.add(new_group)
        session.commit()
    except Exception as e:
        session.rollback()
        log_error(f"Error creating search term group: {str(e)}")

def delete_search_term_group(session, group_id):
    try:
        group = session.query(SearchTermGroup).get(group_id)
        if group:
            # Set group_id to None for all search terms in this group
            session.query(SearchTerm).filter(SearchTerm.group_id == group_id).update({SearchTerm.group_id: None})
            session.delete(group)
    except Exception as e:
        session.rollback()
        log_error(f"Error deleting search term group: {str(e)}")

def email_templates_page():
    st.header("Email Templates")
    with safe_db_session() as session:
        templates = session.query(EmailTemplate).all()
        with st.expander("Create New Template", expanded=False):
            new_template_name = st.text_input("Template Name", key="new_template_name")
            use_ai = st.checkbox("Use AI to generate template", key="use_ai")
            if use_ai:
                ai_prompt = st.text_area("Enter prompt for AI template generation", key="ai_prompt")
                use_kb = st.checkbox("Use Knowledge Base information", key="use_kb")
                if st.button("Generate Template", key="generate_ai_template"):
                    with st.spinner("Generating template..."):
                        kb_info = get_knowledge_base_info(session, get_active_project_id()) if use_kb else None
                        generated_template = generate_or_adjust_email_template(ai_prompt, kb_info)
                        new_template_subject = generated_template.get("subject", "AI Generated Subject")
                        new_template_body = generated_template.get("body", "")

                        if new_template_name:
                            new_template = EmailTemplate(
                                template_name=new_template_name,
                                subject=new_template_subject,
                                body_content=new_template_body,
                                campaign_id=get_active_campaign_id()
                            )
                            session.add(new_template)
                            session.commit()
                            st.success("AI-generated template created and saved!")
                            templates = session.query(EmailTemplate).all()
                        else:
                            st.warning("Please provide a name for the template before generating.")

                        st.subheader("Generated Template")
                        st.text(f"Subject: {new_template_subject}")
                        st.components.v1.html(wrap_email_body(new_template_body), height=400, scrolling=True)
            else:
                new_template_subject = st.text_input("Subject", key="new_template_subject")
                new_template_body = st.text_area("Body Content", height=200, key="new_template_body")

            if st.button("Create Template", key="create_template_button"):
                if all([new_template_name, new_template_subject, new_template_body]):
                    new_template = EmailTemplate(
                        template_name=new_template_name,
                        subject=new_template_subject,
                        body_content=new_template_body,
                        campaign_id=get_active_campaign_id()
                    )
                    session.add(new_template)
                    session.commit()
                    st.success("New template created successfully!")
                    templates = session.query(EmailTemplate).all()
                else:
                    st.warning("Please fill in all fields to create a new template.")

        if templates:
            st.subheader("Existing Templates")
            for template in templates:
                with st.expander(f"Template: {template.template_name}", expanded=False):
                    col1, col2 = st.columns(2)
                    edited_subject = col1.text_input("Subject", value=template.subject, key=f"subject_{template.id}")
                    is_ai_customizable = col2.checkbox("AI Customizable", value=template.is_ai_customizable, key=f"ai_{template.id}")
                    edited_body = st.text_area("Body Content", value=template.body_content, height=200, key=f"body_{template.id}")

                    ai_adjustment_prompt = st.text_area("AI Adjustment Prompt", key=f"ai_prompt_{template.id}", placeholder="E.g., Make it less marketing-like and mention our main features")

                    col3, col4 = st.columns(2)
                    if col3.button("Apply AI Adjustment", key=f"apply_ai_{template.id}"):
                        with st.spinner("Applying AI adjustment..."):
                            kb_info = get_knowledge_base_info(session, get_active_project_id())
                            adjusted_template = generate_or_adjust_email_template(ai_adjustment_prompt, kb_info, current_template=edited_body)
                            edited_subject = adjusted_template.get("subject", edited_subject)
                            edited_body = adjusted_template.get("body", edited_body)
                            st.success("AI adjustment applied. Please review and save changes.")

                    if col4.button("Save Changes", key=f"save_{template.id}"):
                        template.subject = edited_subject
                        template.body_content = edited_body
                        template.is_ai_customizable = is_ai_customizable
                        session.commit()
                        st.success("Template updated successfully!")

                    st.markdown("### Preview")
                    st.text(f"Subject: {edited_subject}")
                    st.components.v1.html(wrap_email_body(edited_body), height=400, scrolling=True)

                    if st.button("Delete Template", key=f"delete_{template.id}"):
                        session.delete(template)
                        session.commit()
                        st.success("Template deleted successfully!")
                        st.rerun()
        else:
            st.info("No email templates found. Create a new template to get started.")

def get_email_preview(session, template_id, from_email, reply_to):
    template = session.query(EmailTemplate).filter_by(id=template_id).first()
    if template:
        wrapped_content = wrap_email_body(template.body_content)
        return wrapped_content
    return "<p>Template not found</p>"

def fetch_all_search_terms(session):
    return session.query(SearchTerm).all()

@functools.lru_cache(maxsize=10)
def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

def get_email_template_by_name(session, template_name):
    return session.query(EmailTemplate).filter_by(template_name=template_name).first()

def bulk_send_page():
    st.title("Bulk Email Sending")
    with safe_db_session() as session:
        templates = fetch_email_templates(session)
        email_settings = fetch_email_settings(session)
        if not templates or not email_settings:
            st.error("No email templates or settings available. Please set them up first.")
            return

        template_option = st.selectbox("Email Template", options=templates, format_func=lambda x: x.split(":")[1].strip())
        template_id = int(template_option.split(":")[0])
        template = session.query(EmailTemplate).filter_by(id=template_id).first()

        col1, col2 = st.columns(2)
        with col1:
            subject = st.text_input("Subject", value=template.subject if template else "")
            email_setting_option = st.selectbox("From Email", options=email_settings, format_func=lambda x: f"{x['name']} ({x['email']})")
            if email_setting_option:
                from_email = email_setting_option['email']
                reply_to = st.text_input("Reply To", email_setting_option['email'])
            else:
                st.error("Selected email setting not found. Please choose a valid email setting.")
                return

        with col2:
            send_option = st.radio("Send to:", ["All Leads", "Specific Email", "Leads from Chosen Search Terms", "Leads from Search Term Groups"])
            specific_email = None
            selected_terms = None
            if send_option == "Specific Email":
                specific_email = st.text_input("Enter email")
            elif send_option == "Leads from Chosen Search Terms":
                search_terms_with_counts = fetch_search_terms_with_lead_count(session)
                selected_terms = st.multiselect("Select Search Terms", options=search_terms_with_counts['Term'].tolist())
                selected_terms = [term.split(" (")[0] for term in selected_terms]
            elif send_option == "Leads from Search Term Groups":
                groups = fetch_search_term_groups(session)
                selected_groups = st.multiselect("Select Search Term Groups", options=groups)
                if selected_groups:
                    group_ids = [int(group.split(':')[0]) for group in selected_groups]
                    selected_terms = fetch_search_terms_for_groups(session, group_ids)

        exclude_previously_contacted = st.checkbox("Exclude Previously Contacted Domains", value=True)

        st.markdown("### Email Preview")
        st.text(f"From: {from_email}\nReply-To: {reply_to}\nSubject: {subject}")
        st.components.v1.html(get_email_preview(session, template_id, from_email, reply_to), height=600, scrolling=True)

        leads = fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted)
        total_leads = len(leads)
        eligible_leads = [lead for lead in leads if lead.get('language', template.language) == template.language]
        contactable_leads = [lead for lead in eligible_leads if not (exclude_previously_contacted and lead.get('domain_contacted', False))]

        st.info(f"Total leads: {total_leads}\n"
                f"Leads matching template language ({template.language}): {len(eligible_leads)}\n"
                f"Leads to be contacted: {len(contactable_leads)}")

        user_settings = get_user_settings()
        enable_email_sending = user_settings['enable_email_sending']

        if st.button("Send Emails", type="primary"):
            if not contactable_leads:
                st.warning("No leads found matching the selected criteria.")
                return
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            log_container = st.empty()
            try:
                for idx, lead in enumerate(contactable_leads):
                    if not is_valid_email(lead['Email']):
                        log_container.warning(f"Skipping invalid email: {lead['Email']}")
                        continue
                    try:
                        response, tracking_id = rate_limited_send_email(session, from_email, lead['Email'], template.subject, template.body_content, reply_to=reply_to)
                        if response:
                            update_log(log_container, f"Sent email to: {lead['Email']}", 'email_sent')
                            save_email_campaign(session, lead['Email'], template.id, 'Sent', datetime.utcnow(), template.subject, response['MessageId'], template.body_content)
                        else:
                            update_log(log_container, f"Failed to send email to: {lead['Email']}", 'error')
                    except Exception as e:
                        update_log(log_container, f"Error sending email to {lead['Email']}: {str(e)}", 'error')
                    progress_bar.progress((idx + 1) / len(contactable_leads))
            except Exception as e:
                log_container.error(f"Error in bulk email sending: {str(e)}")
            finally:
                sent_count = len([res for res in results if res['Status'] == 'Sent'])
                st.success(f"Emails sent successfully to {sent_count} leads.")
                st.subheader("Sending Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                success_rate = (results_df['Status'] == 'Sent').mean()
                st.metric("Email Sending Success Rate", f"{success_rate:.2%}")

def bulk_save_email_campaigns(session: Session, campaigns: List[dict]):
    stmt = insert(EmailCampaign).values(campaigns)
    stmt = stmt.on_conflict_do_update(
        index_elements=['lead_id', 'template_id'],
        set_={
            'status': stmt.excluded.status,
            'sent_at': stmt.excluded.sent_at,
            'customized_subject': stmt.excluded.customized_subject,
            'message_id': stmt.excluded.message_id,
            'customized_content': stmt.excluded.customized_content
        }
    )
    session.execute(stmt)
    session.commit()

def fetch_leads(session, template_id, send_option, specific_email, selected_terms, exclude_previously_contacted):
    try:
        query = session.query(Lead)
        if send_option == "Specific Email":
            query = query.filter(Lead.email == specific_email)
        elif send_option in ["Leads from Chosen Search Terms", "Leads from Search Term Groups"] and selected_terms:
            query = query.join(LeadSource).join(SearchTerm).filter(SearchTerm.term.in_(selected_terms))

        if exclude_previously_contacted:
            subquery = session.query(EmailCampaign.lead_id).filter(EmailCampaign.sent_at.isnot(None)).subquery()
            query = query.outerjoin(subquery, Lead.id == subquery.c.lead_id).filter(subquery.c.lead_id.is_(None))

        return [{"Email": lead.email, "ID": lead.id} for lead in query.all()]
    except Exception as e:
        log_error(f"Error fetching leads: {str(e)}")
        return []

def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
    except Exception as e:
        log_error(f"Error fetching email settings: {e}")
        return []

@functools.lru_cache(maxsize=1)
def fetch_search_terms_with_lead_count(session):
    query = session.query(
        SearchTerm.term,
        func.count(distinct(Lead.id)).label('lead_count'),
        func.count(distinct(EmailCampaign.id)).label('email_count')
    ).join(LeadSource, SearchTerm.id == LeadSource.search_term_id
    ).join(Lead, LeadSource.lead_id == Lead.id
    ).outerjoin(EmailCampaign, Lead.id == EmailCampaign.lead_id
    ).group_by(SearchTerm.term)

    return pd.DataFrame(query.all(), columns=['Term', 'Lead Count', 'Email Count'])

def fetch_search_term_groups(session):
    return [f"{group.id}: {group.name}" for group in session.query(SearchTermGroup).all()]

def fetch_search_terms_for_groups(session, group_ids):
    return [term.term for term in session.query(SearchTerm).filter(SearchTerm.group_id.in_(group_ids)).all()]

def projects_campaigns_page():
    with safe_db_session() as session:
        st.header("Projects and Campaigns")
        st.subheader("Add New Project")
        with st.form("add_project_form"):
            project_name = st.text_input("Project Name")
            if st.form_submit_button("Add Project"):
                if project_name.strip():
                    try:
                        session.add(Project(project_name=project_name, created_at=datetime.utcnow()))
                        session.commit()
                        st.success(f"Project '{project_name}' added successfully.")
                    except SQLAlchemyError as e:
                        st.error(f"Error adding project: {str(e)}")
                else:
                    st.warning("Please enter a project name.")
        st.subheader("Existing Projects and Campaigns")
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
                            session.delete(campaign)
                            session.commit()
                            st.success(f"Campaign {campaign.id} deleted successfully.")
                            st.rerun()
                with st.form(f"add_campaign_form_{project.id}"):
                    campaign_name = st.text_input("Campaign Name", key=f"campaign_name_{project.id}")
                    if st.form_submit_button("Add Campaign", key=f"add_campaign_{project.id}"):
                        if campaign_name.strip():
                            try:
                                session.add(Campaign(campaign_name=campaign_name, project_id=project.id, created_at=datetime.utcnow()))
                                session.commit()
                                st.success(f"Campaign '{campaign_name}' added successfully.")
                            except SQLAlchemyError as e:
                                st.error(f"Error adding campaign: {str(e)}")
                        else:
                            st.warning("Please enter a campaign name.")
        st.subheader("Set Active Project and Campaign")
        project_options = [p.project_name for p in projects]
        if project_options:
            active_project = st.selectbox("Select Active Project", options=project_options, index=0)
            active_project_id = session.query(Project.id).filter_by(project_name=active_project).scalar()
            set_active_project_id(active_project_id)
            active_project_campaigns = session.query(Campaign).filter_by(project_id=active_project_id).all()
            if active_project_campaigns:
                campaign_options = [c.campaign_name for c in active_project_campaigns]
                active_campaign = st.selectbox("Select Active Campaign", options=campaign_options, index=0)
                active_campaign_id = session.query(Campaign.id).filter_by(campaign_name=active_campaign, project_id=active_project_id).scalar()
                set_active_campaign_id(active_campaign_id)
                st.success(f"Active Project: {active_project}, Active Campaign: {active_campaign}")
            else:
                st.warning(f"No campaigns available for {active_project}. Please add a campaign.")
        else:
            st.warning("No projects found. Please add a project first.")

def knowledge_base_page():
    st.title("Knowledge Base")
    with safe_db_session() as session:
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
                except Exception as e:
                    st.error(f"An error occurred while saving the Knowledge Base: {str(e)}")

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
            with safe_db_session() as session:
                base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
                optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)
            if optimized_terms:
                st.session_state.optimized_terms = optimized_terms
                st.success("Search terms optimized successfully!")
                st.subheader("Optimized Search Terms")
                st.write(", ".join(optimized_terms))
            else:
                st.error("Failed to generate optimized search terms. Please try again.")
    if st.button("Start Automation", key="start_automation"):
        st.session_state.update({"automation_status": True, "automation_logs": [], "total_leads_found": 0, "total_emails_sent": 0})
        st.success("Automation started!")
    if st.session_state.get('automation_status', False):
        st.subheader("Automation in Progress")
        progress_bar, log_container, leads_container, analytics_container = st.progress(0), st.empty(), st.empty(), st.empty()
        try:
            with safe_db_session() as session:
                while st.session_state.get('automation_status', False):
                    kb_info = get_knowledge_base_info(session, get_active_project_id())
                    if not kb_info:
                        st.session_state.automation_logs.append("Knowledge Base not found. Skipping cycle.")
                        time.sleep(3600)
                        continue

                    base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
                    optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)

                    new_leads_all = []
                    for term in optimized_terms:
                        results = asyncio.run(manual_search(session, [term], 10))
                        new_leads = [(res['Email'], res['URL']) for res in results['results'] if save_lead(session, res['Email'], url=res['URL'])]
                        new_leads_all.extend(new_leads)

                        if new_leads:
                            template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                            if template:
                                from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                                reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                                logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': email} for email, _ in new_leads])
                                st.session_state.automation_logs.extend(logs)

                    if new_leads_all:
                        leads_df = pd.DataFrame(new_leads_all, columns=['Email', 'URL'])
                        leads_container.dataframe(leads_df, hide_index=True)
                    else:
                        leads_container.info("No new leads found in this cycle.")

                    display_results_and_logs(log_container, st.session_state.get('automation_logs', []), "Latest Logs", "logs")
                    time.sleep(3600)
        except Exception as e:
            st.error(f"An error occurred in the automation process: {str(e)}")

    if not st.session_state.get('automation_status', False) and st.session_state.get('automation_logs'):
        st.subheader("Automation Results")
        st.metric("Total Leads Found", st.session_state.total_leads_found)
        st.metric("Total Emails Sent", st.session_state.total_emails_sent)
        st.subheader("Automation Logs")
        st.text_area("Logs", "\n".join(st.session_state.automation_logs), height=300)
    if 'email_logs' in st.session_state:
        st.subheader("Email Sending Logs")
        df_logs = pd.DataFrame(st.session_state.email_logs)
        st.dataframe(df_logs)
        success_rate = (df_logs['Status'] == 'sent').mean() * 100
        st.metric("Email Sending Success Rate", f"{success_rate:.2f}%")
    st.subheader("Debug Information")
    st.json(st.session_state)
    st.write("Current function:", autoclient_ai_page.__name__)
    st.write("Session state keys:", list(st.session_state.keys()))

def update_search_terms(session, classified_terms):
    for group, terms in classified_terms.items():
        for term in terms:
            existing_term = session.query(SearchTerm).filter_by(term=term, project_id=get_active_project_id()).first()
            if existing_term:
                existing_term.group = group
            else:
                session.add(SearchTerm(term=term, group=group, project_id=get_active_project_id()))
    session.commit()

def update_results_display(results_container, results):
    results_container.markdown(
        f"""
        <style>
        .results-container {{
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: 1rem;
            background-color: rgba(49, 51, 63, 0.1);
        }}
        .result-entry {{
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 0.25rem;
        }}
        </style>
        <div class="results-container">
            <h4>Found Leads ({len(results)})</h4>
            {"".join(f'<div class="result-entry"><strong>{res["Email"]}</strong><br>{res["URL"]}</div>' for res in results[-10:])}
        </div>
        """,
        unsafe_allow_html=True
    )

def automation_control_panel_page():
    st.title("Automation Control Panel")

    col1, col2 = st.columns([2, 1])
    with col1:
        status = "Active" if st.session_state.get('automation_status', False) else "Inactive"
        st.metric("Automation Status", status)
    with col2:
        button_text = "Stop Automation" if st.session_state.get('automation_status', False) else "Start Automation"
        if st.button(button_text, use_container_width=True):
            st.session_state.automation_status = not st.session_state.get('automation_status', False)
            if st.session_state.automation_status:
                st.session_state.automation_logs = []
            st.rerun()

    if st.button("Perform Quick Scan", use_container_width=True):
        with st.spinner("Performing quick scan..."):
            try:
                with safe_db_session() as session:
                    new_leads = session.query(Lead).filter(Lead.is_processed == False).count()
                    session.query(Lead).filter(Lead.is_processed == False).update({Lead.is_processed: True})
                    session.commit()
                    st.success(f"Quick scan completed! Found {new_leads} new leads.")
            except Exception as e:
                st.error(f"An error occurred during quick scan: {str(e)}")

    st.subheader("Real-Time Analytics")
    try:
        with safe_db_session() as session:
            total_leads = session.query(Lead).count()
            emails_sent = session.query(EmailCampaign).count()
            col1, col2 = st.columns(2)
            col1.metric("Total Leads", total_leads)
            col2.metric("Emails Sent", emails_sent)
    except Exception as e:
        st.error(f"An error occurred while displaying analytics: {str(e)}")

    st.subheader("Automation Logs")
    log_container = st.empty()
    display_results_and_logs(log_container, st.session_state.get('automation_logs', []), "Latest Logs", "logs")

    st.subheader("Recently Found Leads")
    leads_container = st.empty()

    if st.session_state.get('automation_status', False):
        st.info("Automation is currently running in the background.")
        try:
            with safe_db_session() as session:
                while st.session_state.get('automation_status', False):
                    kb_info = get_knowledge_base_info(session, get_active_project_id())
                    if not kb_info:
                        st.session_state.automation_logs.append("Knowledge Base not found. Skipping cycle.")
                        time.sleep(3600)
                        continue

                    base_terms = [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]
                    optimized_terms = generate_optimized_search_terms(session, base_terms, kb_info)

                    new_leads_all = []
                    for term in optimized_terms:
                        results = asyncio.run(manual_search(session, [term], 10))
                        new_leads = [(res['Email'], res['URL']) for res in results['results'] if save_lead(session, res['Email'], url=res['URL'])]
                        new_leads_all.extend(new_leads)

                        if new_leads:
                            template = session.query(EmailTemplate).filter_by(project_id=get_active_project_id()).first()
                            if template:
                                from_email = kb_info.get('contact_email') or 'hello@indosy.com'
                                reply_to = kb_info.get('contact_email') or 'eugproductions@gmail.com'
                                logs, sent_count = bulk_send_emails(session, template.id, from_email, reply_to, [{'Email': email} for email, _ in new_leads])
                                st.session_state.automation_logs.extend(logs)

                    if new_leads_all:
                        leads_df = pd.DataFrame(new_leads_all, columns=['Email', 'URL'])
                        leads_container.dataframe(leads_df, hide_index=True)
                    else:
                        leads_container.info("No new leads found in this cycle.")

                    display_results_and_logs(log_container, st.session_state.get('automation_logs', []), "Latest Logs", "logs")
                    time.sleep(3600)
        except Exception as e:
            st.error(f"An error occurred in the automation process: {str(e)}")

@functools.lru_cache(maxsize=10)
def get_knowledge_base_info(session, project_id):
    kb_info = session.query(KnowledgeBase).filter_by(project_id=project_id).first()
    return kb_info.to_dict() if kb_info else None

def generate_optimized_search_terms(session, base_terms, kb_info):
    prompt = f"Generate 5 optimized search terms based on: {', '.join(base_terms)}. Context: {json.dumps(kb_info)}"
    messages = [{"role": "user", "content": prompt}]
    response = openai_chat_completion(messages, function_name="generate_optimized_search_terms")
    if isinstance(response, list):
        return response
    elif isinstance(response, str):
        return response.split('\n')
    else:
        return []

def display_results_and_logs(container, items, title, item_type):
    if not items:
        container.info(f"No {item_type} to display yet.")
        return

    container.markdown(
        f"""
        <style>
        .results-container {{
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 0.25rem;
            padding: 1rem;
        }}
        .result-entry {{
            margin-bottom: 0.5rem;
            padding: 0.5rem;
            border-radius: 0.25rem;
            background-color: rgba(28, 131, 225, 0.1);
        }}
        </style>
        <h4>{title}</h4>
        <div class="results-container">
        {"".join(f'<div class="result-entry">{item}</div>' for item in items[-20:])}
        </div>
        """,
        unsafe_allow_html=True
    )

def get_search_terms(session):
    return [term.term for term in session.query(SearchTerm).filter_by(project_id=get_active_project_id()).all()]

def get_ai_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    return openai_chat_completion(messages)

def fetch_email_settings(session):
    try:
        settings = session.query(EmailSettings).all()
        return [{"id": setting.id, "name": setting.name, "email": setting.email} for setting in settings]
    except Exception as e:
        log_error(f"Error fetching email settings: {e}")
        return []

def wrap_email_body(body_content):
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Email Template</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
        </style>
    </head>
    <body>
        {body_content}
    </body>
    </html>
    """

def fetch_sent_email_campaigns(session):
    try:
        email_campaigns = session.query(EmailCampaign).join(Lead).join(EmailTemplate).options(joinedload(EmailCampaign.lead), joinedload(EmailCampaign.template)).order_by(EmailCampaign.sent_at.desc()).all()
        return pd.DataFrame({
            'ID': [ec.id for ec in email_campaigns],
            'Sent At': [ec.sent_at.strftime("%Y-%m-%d %H:%M:%S") if ec.sent_at else "" for ec in email_campaigns],
            'Email': [ec.lead.email for ec in email_campaigns],
            'Template': [ec.template.template_name for ec in email_campaigns],
            'Subject': [ec.customized_subject or "No subject" for ec in email_campaigns],
            'Content': [ec.customized_content or "No content" for ec in email_campaigns],
            'Status': [ec.status for ec in email_campaigns],
            'Message ID': [ec.message_id or "No message ID" for ec in email_campaigns],
            'Campaign ID': [ec.campaign_id for ec in email_campaigns],
            'Lead Name': [f"{ec.lead.first_name or ''} {ec.lead.last_name or ''}".strip() or "Unknown" for ec in email_campaigns],
            'Lead Company': [ec.lead.company or "Unknown" for ec in email_campaigns]
        })
    except SQLAlchemyError as e:
        log_error(f"Database error in fetch_sent_email_campaigns: {str(e)}")
        return pd.DataFrame()

def view_sent_email_campaigns():
    st.header("Sent Email Campaigns")
    try:
        with safe_db_session() as session:
            email_campaigns = fetch_sent_email_campaigns(session)
        if not email_campaigns.empty:
            st.dataframe(email_campaigns)
            st.subheader("Detailed Content")
            selected_campaign = st.selectbox("Select a campaign to view details", email_campaigns['ID'].tolist())
            if selected_campaign:
                campaign_content = email_campaigns[email_campaigns['ID'] == selected_campaign]['Content'].iloc[0]
                st.text_area("Content", campaign_content if campaign_content else "No content available", height=300)
        else:
            st.info("No sent email campaigns found.")
    except Exception as e:
        st.error(f"An error occurred while fetching sent email campaigns: {str(e)}")
        log_error(f"Error in view_sent_email_campaigns: {str(e)}")

def get_user_settings():
    if 'user_settings' not in st.session_state:
        st.session_state.user_settings = {
            'enable_email_sending': True,
            'ignore_previously_fetched': True,
            'shuffle_keywords_option': True,
            'optimize_english': False,
            'optimize_spanish': False,
            'language': 'EN'
        }
    return st.session_state.user_settings

@sleep_and_retry
@limits(calls=100, period=60)
def rate_limited_api_call(func, *args, **kwargs):
    return func(*args, **kwargs)

def load_leads_progressively(session, page_size=100):
    offset = 0
    while True:
        leads = session.query(Lead).order_by(Lead.id).offset(offset).limit(page_size).all()
        if not leads:
            break
        for lead in leads:
            yield lead
        offset += page_size

def optimized_manual_search(session, search_terms, num_results):
    results = []
    for term in search_terms:
        term_results = auto_perform_optimized_search(session, term, num_results)
        results.extend(term_results)
    return results[:num_results]

def auto_perform_optimized_search(session, term, num_results):
    search_query = session.query(Lead).filter(Lead.email.ilike(f"%{term}%"))

    if session.bind.dialect.name == 'postgresql':
        search_query = search_query.with_hint(Lead, 'USE INDEX (ix_lead_email)')

    search_results = search_query.limit(num_results).all()

    return [
        {
            "id": lead.id,
            "email": lead.email,
            "first_name": lead.first_name,
            "last_name": lead.last_name,
            "company": lead.company,
            "position": lead.position,
            "country": lead.country,
            "industry": lead.industry
        }
        for lead in search_results
    ]

def manual_search_page():
    st.title("Manual Search")

    start_background_process()

    user_settings = get_user_settings()

    with safe_db_session() as session:
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

    col1, col2 = st.columns([2, 1])

    with col1:
        search_terms = st_tags(
            label='Enter search terms:',
            text='Press enter to add more',
            value=recent_search_terms,
            suggestions=['software engineer', 'data scientist', 'product manager'],
            maxtags=10,
            key='search_terms_input'
        )
        num_results = st.slider("Results per term", 1, 500, 10)

    with col2:
        st.subheader("Search Options")
        enable_email_sending = st.checkbox("Enable email sending", value=user_settings['enable_email_sending'])
        ignore_previously_fetched = st.checkbox("Ignore fetched domains", value=user_settings['ignore_previously_fetched'])
        shuffle_keywords_option = st.checkbox("Shuffle Keywords", value=user_settings['shuffle_keywords_option'])
        optimize_english = st.checkbox("Optimize (English)", value=user_settings['optimize_english'])
        optimize_spanish = st.checkbox("Optimize (Spanish)", value=user_settings['optimize_spanish'])
        language = st.selectbox("Select Language", options=["ES", "EN"], index=0 if user_settings['language'] == "ES" else 1)

        one_email_per_url = st.checkbox("Only One Email per URL", value=True)
        one_email_per_domain = st.checkbox("Only One Email per Domain", value=True)

        if enable_email_sending:
            with safe_db_session() as session:
                email_templates = fetch_email_templates(session)
                email_settings = fetch_email_settings(session)

            if not email_templates or not email_settings:
                st.error("No email templates or settings available. Please set them up first.")
                return

            col3, col4 = st.columns(2)
            with col3:
                email_template = st.selectbox("Email template", options=email_templates, format_func=lambda x: x.split(":")[1].strip())
            with col4:
                email_setting_option = st.selectbox("From Email", options=email_settings, format_func=lambda x: f"{x['name']} ({x['email']})")
                if email_setting_option:
                    from_email = email_setting_option['email']
                    reply_to = st.text_input("Reply To", email_setting_option['email'])
                else:
                    st.error("No email setting selected. Please select an email setting.")
                    return

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Run Manual Search", use_container_width=True):
            with safe_db_session() as session:
                job_enqueued = enqueue_job(session, 'manual_search', {
                    'search_terms': search_terms,
                    'num_results': num_results,
                    'ignore_previously_fetched': ignore_previously_fetched,
                    'optimize_english': optimize_english,
                    'optimize_spanish': optimize_spanish,
                    'shuffle_keywords_option': shuffle_keywords_option,
                    'language': language,
                    'enable_email_sending': enable_email_sending,
                    'from_email': from_email,
                    'reply_to': reply_to,
                    'email_template': email_template,
                    'one_email_per_url': one_email_per_url,
                    'one_email_per_domain': one_email_per_domain
                })
                if job_enqueued:
                    st.success("Manual search job enqueued successfully!")
                else:
                    st.warning("A job is already running. Please wait for it to complete.")

            with safe_db_session() as session:
                state = get_background_state()
                if state['is_running'] and state['job_type'] == 'manual_search':
                    st.info(f"Manual search in progress. Current term: {state['current_term']}")
                    st.progress(state['job_progress'])

    with col2:
        if st.button("Pause/Resume Background Search", use_container_width=True):
            with safe_db_session() as session:
                state = get_background_state()
                if state['is_running']:
                    pause_background_search()
                    st.success("Background search paused.")
                else:
                    resume_background_search()
                    st.success("Background search resumed.")
    with col3:
        if st.button("Stop Background Search", use_container_width=True):
            stop_background_search()
            st.success("Background search stopped.")

    st.subheader("Latest Leads")
    st.table(pd.DataFrame(latest_leads_data, columns=["Email", "Company", "Created At"]))

    st.subheader("Latest Email Campaigns")
    st.table(pd.DataFrame(latest_campaigns_data, columns=["Email", "Template", "Sent At", "Status"]))

def main():
    try:
        pages = {
            "🔍 Manual Search": manual_search_page,
            "📦 Bulk Send": bulk_send_page,
            "👥 View Leads": view_leads_page,
            "🔑 Search Terms": search_terms_page,
            "✉️ Email Templates": email_templates_page,
            "🚀 Projects & Campaigns": projects_campaigns_page,
            "📚 Knowledge Base": knowledge_base_page,
            "🤖 AutoclientAI": autoclient_ai_page,
            "⚙️ Automation Control": automation_control_panel_page,
            "📨 Email Logs": view_campaign_logs,
            "🔄 Settings": settings_page,
            "📨 Sent Campaigns": view_sent_email_campaigns
        }

        with st.sidebar:
            selected = option_menu(
                menu_title="Navigation",
                options=list(pages.keys()),
                icons=["search", "send", "people", "key", "envelope", "folder", "book", "robot", "gear", "list-check", "gear", "envelope-open"],
                menu_icon="cast",
                default_index=0
            )

        if selected in pages:
            pages[selected]()
        else:
            st.error(f"Selected page '{selected}' not found.")
            

        st.sidebar.markdown("---")
        st.sidebar.info("© 2024 AutoclientAI. All rights reserved.")

    except Exception as e:
        log_error(f"An unexpected error occurred in the main function: {str(e)}")
        st.error("An unexpected error occurred. Please try refreshing the page or contact support if the issue persists.")
        st.exception(e)

    auto_refresh()

if __name__ == "__main__":
    main()







