#!/bin/bash

# Reset database schema first
python reset_schema.py

# Start the background worker
python background_worker.py &

# Start the Streamlit app
streamlit run streamlit_app.py