#!/bin/bash
# Set the Flask app entry point (replace "app.py" with your app's file)
export FLASK_APP=app.py
# Run the Flask app on all network interfaces (required for Hugging Face spaces)
flask run --host=0.0.0.0 --port=7860