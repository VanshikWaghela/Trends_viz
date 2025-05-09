#!/usr/bin/env bash
# This script is executed by Render.com during deployment

# Make the script exit if any command fails
set -e

# Install Python dependencies
pip install -r requirements.txt

# Add any other build steps here if needed
