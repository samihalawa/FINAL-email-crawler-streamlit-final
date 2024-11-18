Database Schema Context:
```python
class Campaign(Base):
    loop_automation = Column(Boolean, default=False)
    max_emails_per_group = Column(BigInteger, default=40)
    loop_interval = Column(BigInteger, default=60)
```

The automation settings are stored in the Campaign model. The UI and database schema from streamlit_app.py are being maintained exactly in the Node.js version. 