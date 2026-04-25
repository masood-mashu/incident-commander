#!/bin/bash

# Start the FastAPI OpenEnv server in the background on port 8000
uvicorn incident_commander.server.app:app --host 0.0.0.0 --port 8000 &

# Start the Gradio app in the foreground on port 7860
python app.py
