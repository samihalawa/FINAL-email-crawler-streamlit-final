#!/bin/bash

# Start the background worker
python background_worker.py &

# Start the Streamlit app
streamlit run streamlit_app.py 