# Email Lead Finder

A production-ready Streamlit application for finding and managing email leads with automated outreach capabilities.

## Features

- 🔍 Advanced email search with domain filtering
- 📧 Automated email outreach with tracking
- 📊 Real-time monitoring and metrics
- 🔄 Background processing with worker threads
- 🛡️ Rate limiting and error handling
- 📈 Performance monitoring with Prometheus
- 🐛 Error tracking with Sentry
- 🔒 Secure credential management
- 📝 Comprehensive logging

## Prerequisites

- Python 3.8+
- PostgreSQL database
- SMTP server or AWS SES account for email sending
- (Optional) Sentry account for error tracking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/email-lead-finder.git
cd email-lead-finder
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```env
# Database
SUPABASE_DB_HOST=your_db_host
SUPABASE_DB_NAME=your_db_name
SUPABASE_DB_USER=your_db_user
SUPABASE_DB_PASSWORD=your_db_password
SUPABASE_DB_PORT=5432

# Error Tracking
SENTRY_DSN=your_sentry_dsn
ENVIRONMENT=production

# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Start the application:
```bash
streamlit run streamlit_app_BACKGROUND_PROCESS_ADDED.py
```

2. Access the web interface at `http://localhost:8501`

3. Configure email settings:
   - Navigate to the Settings page
   - Add SMTP or AWS SES credentials
   - Configure general settings

4. Start searching:
   - Go to Manual Search
   - Enter search terms
   - Configure search options
   - Start the search process

5. Monitor progress:
   - View real-time logs
   - Check system metrics
   - Monitor email sending status

## Monitoring

The application includes comprehensive monitoring:

- Prometheus metrics at `http://localhost:8000`
- System status in the Monitoring page
- Real-time logs in the application
- Error tracking in Sentry

## Production Deployment

For production deployment:

1. Use a production-grade WSGI server:
```bash
pip install waitress
waitress-serve --port=8501 streamlit_app_BACKGROUND_PROCESS_ADDED:app
```

2. Set up a reverse proxy (e.g., Nginx) with SSL

3. Configure environment variables for production

4. Set up monitoring and alerting

## Security Considerations

- All credentials are stored securely in the database
- Rate limiting prevents abuse
- Email tracking respects privacy
- Error handling prevents data leaks
- Database connections are pooled and managed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.
