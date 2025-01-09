#!/bin/bash

# Reset database schema first
python reset_db.py

# Start the Streamlit app
streamlit run streamlit_app.py