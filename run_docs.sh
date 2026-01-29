#!/bin/bash
# Simple script to run the Kitsune documentation site locally

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  Kitsune Documentation Server${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo -e "${YELLOW}MkDocs not found. Installing documentation dependencies...${NC}"
    pip install -e ".[docs]"
    echo ""
fi

echo -e "${GREEN}Starting documentation server...${NC}"
echo -e "📚 Documentation will be available at: ${BLUE}http://127.0.0.1:8000${NC}"
echo -e "⚡ Live reload enabled - changes will update automatically"
echo -e ""
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop the server"
echo ""
echo -e "${BLUE}========================================${NC}"
echo ""

# Start the server
mkdocs serve
