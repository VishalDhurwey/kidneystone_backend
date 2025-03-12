#!/bin/bash

# Install system dependencies (like libGL for OpenCV)
apt-get update
apt-get install -y libgl1 libglib2.0-0

# Run your Flask app
python app.py
