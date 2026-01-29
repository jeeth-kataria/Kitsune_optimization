#!/bin/bash
# Documentation management script

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  serve      - Serve documentation locally with live reload"
    echo "  build      - Build static documentation site"
    echo "  check      - Build with strict mode to check for errors"
    echo "  deploy     - Deploy to GitHub Pages"
    echo "  clean      - Clean build artifacts"
    echo "  install    - Install documentation dependencies"
    echo ""
}

check_deps() {
    if ! command -v mkdocs &> /dev/null; then
        echo -e "${RED}Error: mkdocs not found${NC}"
        echo "Run: $0 install"
        exit 1
    fi
}

cmd_serve() {
    echo -e "${GREEN}Starting documentation server...${NC}"
    cd "$PROJECT_ROOT"
    mkdocs serve
}

cmd_build() {
    echo -e "${GREEN}Building documentation...${NC}"
    cd "$PROJECT_ROOT"
    mkdocs build
    echo -e "${GREEN}Done! Output in site/${NC}"
}

cmd_check() {
    echo -e "${GREEN}Building documentation with strict mode...${NC}"
    cd "$PROJECT_ROOT"
    if mkdocs build --strict; then
        echo -e "${GREEN}✓ No errors found${NC}"
    else
        echo -e "${RED}✗ Build failed${NC}"
        exit 1
    fi
}

cmd_deploy() {
    echo -e "${YELLOW}Deploying to GitHub Pages...${NC}"
    cd "$PROJECT_ROOT"
    mkdocs gh-deploy --force
    echo -e "${GREEN}Done!${NC}"
}

cmd_clean() {
    echo -e "${GREEN}Cleaning build artifacts...${NC}"
    cd "$PROJECT_ROOT"
    rm -rf site/
    echo -e "${GREEN}Done!${NC}"
}

cmd_install() {
    echo -e "${GREEN}Installing documentation dependencies...${NC}"
    cd "$PROJECT_ROOT"
    pip install -e ".[docs]"
    echo -e "${GREEN}Done!${NC}"
}

# Main command dispatcher
case "${1:-}" in
    serve)
        check_deps
        cmd_serve
        ;;
    build)
        check_deps
        cmd_build
        ;;
    check)
        check_deps
        cmd_check
        ;;
    deploy)
        check_deps
        cmd_deploy
        ;;
    clean)
        cmd_clean
        ;;
    install)
        cmd_install
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        if [ -z "${1:-}" ]; then
            echo -e "${RED}Error: No command specified${NC}"
        else
            echo -e "${RED}Error: Unknown command '$1'${NC}"
        fi
        echo ""
        print_usage
        exit 1
        ;;
esac
