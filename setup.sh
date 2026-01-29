#!/bin/bash
set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}LinuxWhisper Setup${NC}"

# 1. Check for apt (Debian/Ubuntu)
if ! command -v apt &> /dev/null; then
    echo "Error: 'apt' not found. This script supports Debian/Ubuntu based systems."
    exit 1
fi

# 2. Install System Dependencies
echo -e "${BLUE}Installing system packages (password may be required)...${NC}"
sudo apt update
sudo apt install -y libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev \
                    gir1.2-gtk-3.0 gir1.2-ayatanaappindicator3-0.1 gir1.2-webkit2-4.1 \
                    xdotool gnome-screenshot

# 3. Check for uv, install if not present
if ! command -v uv &> /dev/null; then
    echo -e "${BLUE}Installing uv package manager...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the environment to make uv available
    export PATH="$HOME/.local/bin:$PATH"
fi

# 4. Sync Python dependencies using uv
echo -e "${BLUE}Syncing Python dependencies with uv...${NC}"
uv sync

# 5. Success Message
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "To run LinuxWhisper:"
echo "  1. Set your API key: export GROQ_API_KEY='your-key'"
echo "  2. Run: uv run linuxwhisper.py"
echo ""
