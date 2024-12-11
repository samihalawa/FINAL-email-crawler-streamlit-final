# AutoclientAI with Celery

A Streamlit application for lead generation with background task processing using Celery.

## Project Structure

```
.
├── streamlit_app.py    # Main Streamlit application
├── worker_utils.py     # Celery worker utilities
├── config.py          # Configuration settings
├── tasks.py           # Celery task definitions
├── worker.py         # Celery worker entry point
├── requirements.txt   # Project dependencies
├── tests/            # Test directory
│   ├── conftest.py
│   ├── test_integration.py
│   └── test_utils.py
├── pyproject.toml    # Project configuration
├── pytest.ini       # PyTest configuration
└── README.md         # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Redis server:
```bash
redis-server
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

## Development

- Run tests: `pytest`
- Format code: `black .`
- Check types: `mypy .`

## License

MIT License - see LICENSE file for details.
