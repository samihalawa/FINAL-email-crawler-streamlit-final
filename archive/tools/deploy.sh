#!/bin/bash

# Exit on error
set -e

# Configuration
APP_NAME="email-crawler"
APP_DIR="/opt/$APP_NAME"
LOG_DIR="/var/log/$APP_NAME"
SERVICE_NAME="$APP_NAME.service"
PYTHON_VERSION="3.11"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root"
    exit 1
fi

echo "Starting deployment of $APP_NAME..."

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y python$PYTHON_VERSION python$PYTHON_VERSION-venv python3-pip postgresql-client

# Create application directory
echo "Creating application directory..."
mkdir -p $APP_DIR
mkdir -p $LOG_DIR

# Set permissions
echo "Setting permissions..."
chown -R www-data:www-data $APP_DIR
chown -R www-data:www-data $LOG_DIR
chmod 755 $APP_DIR
chmod 755 $LOG_DIR

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
python$PYTHON_VERSION -m venv $APP_DIR/venv
source $APP_DIR/venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy application files
echo "Copying application files..."
cp -r . $APP_DIR/
cp $SERVICE_NAME /etc/systemd/system/

# Set up environment file
if [ ! -f "$APP_DIR/.env" ]; then
    echo "Creating .env file..."
    cp .env.example $APP_DIR/.env
    echo "Please update $APP_DIR/.env with your configuration"
fi

# Reload systemd and enable service
echo "Setting up systemd service..."
systemctl daemon-reload
systemctl enable $SERVICE_NAME
systemctl start $SERVICE_NAME

echo "Checking service status..."
systemctl status $SERVICE_NAME

echo "Deployment completed successfully!"
echo "Logs are available at: $LOG_DIR"
echo "Service status: systemctl status $SERVICE_NAME"
echo "View logs: journalctl -u $SERVICE_NAME" 