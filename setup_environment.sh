#!/bin/bash
# OpenPI Franka Control Environment Setup Script
# This script installs all required dependencies for running OpenPI with Franka robot

set -e  # Exit on error

echo "========================================"
echo "OpenPI Franka Control Setup"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${YELLOW}Step 1/7: Checking conda environment${NC}"
if [[ -z "$CONDA_DEFAULT_ENV" ]] || [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
    echo -e "${RED}Error: Please activate the 'franka' conda environment first${NC}"
    echo "Run: conda activate franka"
    exit 1
fi
echo -e "${GREEN}✓ Conda environment: $CONDA_DEFAULT_ENV${NC}"
echo ""

echo -e "${YELLOW}Step 2/7: Installing Python dependencies${NC}"
pip install zerorpc==0.6.3
pip install "typing_extensions>=4.5.0"
pip install websockets>=10.0
pip install opencv-python==4.8.0.74
echo -e "${GREEN}✓ Python dependencies installed${NC}"
echo ""

echo -e "${YELLOW}Step 3/7: Installing openpi-client${NC}"
OPENPI_CLIENT_PATH="$HOME/thomas/openpi/packages/openpi-client"
if [ -d "$OPENPI_CLIENT_PATH" ]; then
    cd "$OPENPI_CLIENT_PATH"
    uv pip install -e . --system
    echo -e "${GREEN}✓ openpi-client installed from: $OPENPI_CLIENT_PATH${NC}"
else
    echo -e "${RED}Warning: openpi-client not found at $OPENPI_CLIENT_PATH${NC}"
    echo "You'll need to install it manually later"
fi
cd "$SCRIPT_DIR"
echo ""

echo -e "${YELLOW}Step 4/7: Checking ZED SDK installation${NC}"
if [ -d "/usr/local/zed" ]; then
    echo -e "${GREEN}✓ ZED SDK already installed${NC}"
else
    echo -e "${YELLOW}Installing ZED SDK 4.0 for Ubuntu 20.04...${NC}"
    
    # Download ZED SDK 4.0 if not already present
    ZED_INSTALLER="$HOME/Downloads/ZED_SDK_4.0_Ubuntu20.run"
    if [ ! -f "$ZED_INSTALLER" ]; then
        echo "Downloading ZED SDK 4.0 (1.2 GB)..."
        cd "$HOME/Downloads"
        wget --no-check-certificate \
            https://download.stereolabs.com/zedsdk/4.0/cu121/ubuntu20 \
            -O ZED_SDK_4.0_Ubuntu20.run
    fi
    
    # Install ZED SDK
    echo "Installing ZED SDK (requires sudo password)..."
    chmod +x "$ZED_INSTALLER"
    "$ZED_INSTALLER" -- --silent skip_tools skip_cuda
    
    echo -e "${GREEN}✓ ZED SDK installed${NC}"
fi
echo ""

echo -e "${YELLOW}Step 5/7: Installing pyzed for Python 3.10${NC}"
# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    echo -e "${RED}Warning: Expected Python 3.10, but found $PYTHON_VERSION${NC}"
    echo "pyzed installation may fail"
fi

# Download and install pyzed
PYZED_WHEEL="$HOME/pyzed-4.0-cp310-cp310-linux_x86_64.whl"
if [ ! -f "$PYZED_WHEEL" ]; then
    echo "Downloading pyzed wheel..."
    wget --no-check-certificate \
        https://download.stereolabs.com/zedsdk/4.0/whl/linux_x86_64/pyzed-4.0-cp310-cp310-linux_x86_64.whl \
        -O "$PYZED_WHEEL"
fi

pip install "$PYZED_WHEEL"
echo -e "${GREEN}✓ pyzed installed${NC}"
echo ""

echo -e "${YELLOW}Step 6/7: Setting up USB permissions for ZED cameras${NC}"
# Add user to video group
sudo usermod -aG video $USER

# Create udev rules for ZED cameras
echo "Creating udev rules (requires sudo password)..."
sudo sh -c 'echo "SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"2b03\", MODE=\"0666\"" > /etc/udev/rules.d/99-zed.rules'
sudo udevadm control --reload-rules
sudo udevadm trigger
echo -e "${GREEN}✓ USB permissions configured${NC}"
echo -e "${YELLOW}Note: You may need to logout/login for video group changes to take effect${NC}"
echo ""

echo -e "${YELLOW}Step 7/7: Cleaning UV cache to free disk space${NC}"
uv cache clean
echo -e "${GREEN}✓ UV cache cleaned${NC}"
echo ""

# Verify installations
echo "========================================"
echo "Verifying installations..."
echo "========================================"

echo -n "Checking openpi_client... "
python -c "from openpi_client.websocket_client_policy import WebsocketClientPolicy" 2>/dev/null && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"

echo -n "Checking zerorpc... "
python -c "import zerorpc" 2>/dev/null && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"

echo -n "Checking opencv... "
python -c "import cv2; print('✓' if hasattr(cv2, 'cvtColor') else '✗')"

echo -n "Checking pyzed... "
python -c "import pyzed.sl as sl; sl.Camera()" 2>/dev/null && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"

echo -n "Checking ZED cameras... "
lsusb | grep -q "2b03" && echo -e "${GREEN}✓ (cameras detected)${NC}" || echo -e "${YELLOW}⚠ (no cameras detected)${NC}"

echo ""
echo "========================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. If this is your first time: logout and login for group permissions"
echo "2. Activate franka environment: conda activate franka"
echo "3. Run your script: cd openpi_inference && python openpi_bridge_velocity.py"
echo ""
