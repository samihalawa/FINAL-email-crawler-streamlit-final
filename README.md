# AutoclientAI - Lead Generation AI Application

A powerful Streamlit-based application for automated lead generation, email campaigns, and customer outreach.

## 🚀 Features

- Manual and automated lead search
- Email campaign management
- Search term organization and tracking
- Email template management
- Project and campaign organization
- Knowledge base integration
- Automated client outreach
- Email tracking and analytics
- Settings management

## 📋 Requirements

- Python 3.8+
- PostgreSQL database (Supabase)
- OpenAI API access
- AWS SES (optional for email sending)
- SMTP server (optional alternative to AWS SES)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autoclientai.git
cd autoclientai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:
```env
# Database Configuration
SUPABASE_DB_HOST=your_host
SUPABASE_DB_NAME=your_db_name
SUPABASE_DB_USER=your_username
SUPABASE_DB_PASSWORD=your_password
SUPABASE_DB_PORT=5432

# OpenAI Configuration
OPENAI_API_KEY=your_api_key

# AWS Configuration (optional)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region

# Email Configuration (required for email functionality)
SMTP_SERVER=your_smtp_server
SMTP_PORT=587
SMTP_USERNAME=your_username
SMTP_PASSWORD=your_password

# Optional Analytics Configuration
ENABLE_ANALYTICS=true
ANALYTICS_KEY=your_analytics_key
```

## 🗄️ Database Schema

The application uses the following main tables:
- Projects
- Campaigns
- SearchTerms
- EmailTemplates
- Settings
- EmailSettings
- Leads
- EmailLogs

## 📊 Application Structure

```
streamlit_app.py
├── Database Models
│   ├── Project
│   ├── Campaign
│   ├── SearchTerm
│   ├── EmailTemplate
│   └── Settings
│
├── Core Pages
│   ├── manual_search_page()
│   ├── bulk_send_page()
│   ├── view_leads_page()
│   ├── search_terms_page()
│   ├── email_templates_page()
│   ├── projects_campaigns_page()
│   ├── knowledge_base_page()
│   ├── autoclient_ai_page()
│   ├── automation_control_panel_page()
│   ├── view_campaign_logs()
│   └── settings_page()
│
└── Utility Functions
    ├── db_session()
    ├── send_email_ses()
    ├── validate_settings()
    └── update_results_display()
```

## 🚀 Running the Application

1. Start the application:
```bash
streamlit run streamlit_app.py
```

2. Navigate to `http://localhost:8501` in your browser

3. Configure settings:
   - Add OpenAI API credentials
   - Configure email settings (SMTP or AWS SES)
   - Set up projects and campaigns

## 📝 Usage Flow

1. **Project Setup**
   - Create a new project
   - Add campaigns within the project

2. **Search Configuration**
   - Add search terms
   - Configure search parameters
   - Set up email templates

3. **Lead Generation**
   - Use manual search or automated AI search
   - Review and manage leads
   - Execute email campaigns

4. **Monitoring**
   - Track email campaign performance
   - View lead statistics
   - Monitor automation logs

## 🔒 Security Considerations

- All sensitive credentials are stored in environment variables
- Database connections use secure SSL
- Email sending supports both SMTP and AWS SES
- API keys are encrypted in the database

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Open a pull request

## 📄 License

© 2024 AutoclientAI. All rights reserved.

## 📝 Function Relationship Diagram

```
Core Application Flow:
main() → Navigation
├── Data Management
│   ├── manual_search_page() ←→ fetch_search_terms_with_lead_count()
│   ├── bulk_send_page() ←→ send_email_ses()
│   └── view_leads_page() ←→ db_session()
│
├── Configuration
│   ├── settings_page() ←→ validate_settings()
│   ├── email_templates_page()
│   └── search_terms_page()
│
├── Project Management
│   ├── projects_campaigns_page()
│   └── knowledge_base_page()
│
└── Automation
    ├── autoclient_ai_page() ←→ ai_automation_loop()
    ├── automation_control_panel_page()
    └── view_campaign_logs()
```

## 📝 Initial Setup

1. Create database tables:
```bash
python setup_database.py
```

2. Configure email settings:
   - SMTP or AWS SES configuration required
   - Test email configuration before deployment

3. Initialize OpenAI:
   - Verify API key permissions
   - Test model access
