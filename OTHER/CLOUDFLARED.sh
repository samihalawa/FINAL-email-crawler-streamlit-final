#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to check if cloudflared is installed
check_cloudflared() {
    if ! command -v cloudflared &> /dev/null; then
        echo -e "${RED}cloudflared not found. Installing...${NC}"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install cloudflared
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
            sudo dpkg -i cloudflared.deb
            rm cloudflared.deb
        else
            echo -e "${RED}Please install cloudflared manually${NC}"
            exit 1
        fi
    fi
}

# Function to login to Cloudflare
login_cloudflare() {
    echo -e "${BLUE}Logging into Cloudflare...${NC}"
    cloudflared tunnel login
}

# Function to create and configure tunnel
setup_tunnel() {
    local tunnel_name="ssh-tunnel-$(date +%s)"
    local domain="$1"
    local port="$2"

    # Create tunnel
    echo -e "${BLUE}Creating tunnel: $tunnel_name${NC}"
    tunnel_id=$(cloudflared tunnel create "$tunnel_name" | grep -o '[a-f0-9]\{8\}-[a-f0-9]\{4\}-[a-f0-9]\{4\}-[a-f0-9]\{4\}-[a-f0-9]\{12\}')
    
    if [ -z "$tunnel_id" ]; then
        echo -e "${RED}Failed to create tunnel${NC}"
        exit 1
    fi

    # Create config file
    config_dir="$HOME/.cloudflared"
    mkdir -p "$config_dir"
    
    cat > "$config_dir/config.yml" << EOF
tunnel: ${tunnel_id}
credentials-file: ${config_dir}/${tunnel_id}.json
ingress:
  - hostname: ${domain}
    service: ssh://localhost:${port}
  - service: http_status:404
EOF

    # Route traffic
    echo -e "${BLUE}Routing traffic to tunnel...${NC}"
    cloudflared tunnel route dns "$tunnel_id" "$domain"

    # Create systemd service for auto-start (Linux only)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        create_systemd_service "$tunnel_id"
    fi

    # Start tunnel
    echo -e "${GREEN}Starting tunnel...${NC}"
    cloudflared tunnel run "$tunnel_id" &

    # Save tunnel info
    echo -e "\n${GREEN}=== Tunnel Information ===${NC}"
    echo "Tunnel Name: $tunnel_name"
    echo "Tunnel ID: $tunnel_id"
    echo "Domain: $domain"
    echo "Port: $port"
    echo -e "\nTo connect via SSH:"
    echo "ssh samihalawa@$domain"
    
    # Save to a local file
    echo -e "\n${BLUE}Saving tunnel info to ~/cloudflared-tunnel-info.txt${NC}"
    {
        echo "Tunnel Name: $tunnel_name"
        echo "Tunnel ID: $tunnel_id"
        echo "Domain: $domain"
        echo "Port: $port"
        echo "SSH Command: ssh samihalawa@$domain"
        echo "Config File: $config_dir/config.yml"
    } > ~/cloudflared-tunnel-info.txt
}

# Function to create systemd service (Linux only)
create_systemd_service() {
    local tunnel_id="$1"
    
    sudo tee /etc/systemd/system/cloudflared-ssh.service << EOF
[Unit]
Description=Cloudflare Tunnel for SSH
After=network.target

[Service]
Type=simple
User=$USER
ExecStart=$(which cloudflared) tunnel run ${tunnel_id}
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.default.target
EOF

    sudo systemctl enable cloudflared-ssh
    sudo systemctl start cloudflared-ssh
}

# Main script
echo -e "${BLUE}SSH Tunnel Setup Script${NC}"
echo "================================"

# Check and install cloudflared
check_cloudflared

# Get user input
read -p "Enter your domain (e.g., ssh.mubago.com): " DOMAIN
read -p "Enter SSH port (default: 22): " PORT
PORT=${PORT:-22}

# Login and setup
login_cloudflare
setup_tunnel "$DOMAIN" "$PORT"

echo -e "\n${GREEN}Setup complete! Your tunnel is running.${NC}"
echo "Configuration and credentials are saved in ~/.cloudflared/"
echo "Tunnel information is saved in ~/cloudflared-tunnel-info.txt"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "\nTunnel service is installed and will start automatically on boot"
    echo "Use 'sudo systemctl status cloudflared-ssh' to check service status"
fi

echo -e "\n${BLUE}To connect from your iPhone:${NC}"
echo "1. Install Cloudflared on your iPhone"
echo "2. Use any SSH client (like Termius)"
echo "3. Connect to: $DOMAIN"
echo "4. Username: samihalawa"