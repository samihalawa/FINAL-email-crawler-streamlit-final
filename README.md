# AutoClient.ai

A powerful automated lead generation and email outreach platform built with Python and NiceGUI.

## Features

### Lead Generation
- Multi-source search (Google, LinkedIn, etc.)
- AI-powered lead qualification
- Automated data enrichment
- Smart duplicate detection
- Company and contact information extraction

### Email Outreach
- SMTP and AWS SES support
- AI-optimized email templates
- Personalized content generation
- Open and click tracking
- Bounce and complaint handling

### Automation
- Project-based automation
- Parallel task processing
- Smart scheduling
- Auto-pause on high bounce rates
- Resource optimization

### Analytics
- Real-time campaign monitoring
- Performance analytics
- AI-powered insights
- Detailed email logs
- Export capabilities

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/autoclient-ai.git
cd autoclient-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r src/requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your settings
```

## Configuration

### Email Settings
- SMTP configuration
- AWS SES credentials
- Email templates
- Sending limits

### Search Settings
- API keys
- Search sources
- Excluded domains
- Rate limits

### AI Settings
- OpenAI API key
- Model selection
- Temperature settings

## Usage

1. Start the application:
```bash
cd src
python main.py
```

2. Access the web interface:
```
http://localhost:8080
```

3. Create a project:
- Configure search terms
- Set up email templates
- Define automation rules

4. Start automation:
- Enable project automation
- Monitor progress
- View analytics

## Project Structure

```
src/
├── core/
│   ├── auth.py         # Authentication
│   ├── background.py   # Task management
│   ├── config.py       # Configuration
│   ├── database.py     # Database models
│   └── logging.py      # Logging setup
├── pages/
│   ├── automation_control.py
│   ├── bulk_send.py
│   ├── campaigns.py
│   ├── email_logs.py
│   ├── email_templates.py
│   ├── knowledge_base.py
│   ├── leads.py
│   ├── manual_search.py
│   ├── projects.py
│   ├── search_terms.py
│   └── settings.py
├── services/
│   ├── ai.py          # AI integration
│   ├── email.py       # Email handling
│   └── search.py      # Search functionality
├── utils/
│   └── validation.py  # Data validation
├── main.py            # Application entry
└── requirements.txt   # Dependencies
```

## Performance Optimization

The application is optimized for maximum performance:

1. **Parallel Processing**
   - Multi-threaded search
   - Concurrent email sending
   - Parallel data processing

2. **Resource Management**
   - Connection pooling
   - Memory optimization
   - Cache management

3. **Database Optimization**
   - Indexed queries
   - Efficient data structures
   - Batch processing

4. **Network Optimization**
   - Connection reuse
   - Request batching
   - Response streaming

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 