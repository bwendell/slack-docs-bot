#!/usr/bin/env bash
#
# Ollama Setup Script for ai-docs-bot
#
# Usage:
#   ./scripts/ollama-setup.sh [command]
#
# Commands:
#   install   - Install Ollama (requires sudo)
#   pull      - Pull the configured model
#   start     - Start Ollama server
#   stop      - Stop Ollama server
#   status    - Check Ollama status
#   test      - Run a quick test query
#   setup     - Full setup: install, pull, start (default)
#
# Environment Variables:
#   OLLAMA_MODEL      - Model to pull/use (default: llama3.2)
#   OLLAMA_BASE_URL   - Ollama API URL (default: http://localhost:11434)
#

set -e

# Configuration
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Ollama is installed
is_ollama_installed() {
    command -v ollama &> /dev/null
}

# Check if Ollama server is running
is_ollama_running() {
    curl -s "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1
}

# Check if model is available
is_model_available() {
    local models
    models=$(curl -s "${OLLAMA_BASE_URL}/api/tags" 2>/dev/null | grep -o "\"name\":\"[^\"]*\"" | grep -E "\"${OLLAMA_MODEL}(:latest)?\"" || true)
    [[ -n "$models" ]]
}

cmd_install() {
    log_info "Installing Ollama..."
    
    if is_ollama_installed; then
        local version
        version=$(ollama --version 2>/dev/null | head -1 || echo "unknown")
        log_warn "Ollama is already installed: $version"
        return 0
    fi
    
    # Download and run installer
    curl -fsSL https://ollama.com/install.sh | sh
    
    if is_ollama_installed; then
        log_info "Ollama installed successfully!"
        ollama --version
    else
        log_error "Ollama installation failed"
        exit 1
    fi
}

cmd_pull() {
    log_info "Pulling model: ${OLLAMA_MODEL}..."
    
    if ! is_ollama_installed; then
        log_error "Ollama is not installed. Run: $0 install"
        exit 1
    fi
    
    # Ensure server is running for pull
    if ! is_ollama_running; then
        log_info "Starting Ollama server for model pull..."
        cmd_start
        sleep 2
    fi
    
    ollama pull "${OLLAMA_MODEL}"
    log_info "Model ${OLLAMA_MODEL} pulled successfully!"
}

cmd_start() {
    log_info "Starting Ollama server..."
    
    if ! is_ollama_installed; then
        log_error "Ollama is not installed. Run: $0 install"
        exit 1
    fi
    
    if is_ollama_running; then
        log_warn "Ollama server is already running"
        return 0
    fi
    
    # Start server in background
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    
    # Wait for server to be ready
    log_info "Waiting for Ollama server to start..."
    for i in {1..30}; do
        if is_ollama_running; then
            log_info "Ollama server started successfully!"
            return 0
        fi
        sleep 1
    done
    
    log_error "Ollama server failed to start. Check /tmp/ollama.log for details."
    exit 1
}

cmd_stop() {
    log_info "Stopping Ollama server..."
    
    if ! is_ollama_running; then
        log_warn "Ollama server is not running"
        return 0
    fi
    
    # Find and kill Ollama processes
    pkill -f "ollama serve" 2>/dev/null || true
    
    sleep 2
    
    if is_ollama_running; then
        log_error "Failed to stop Ollama server"
        exit 1
    fi
    
    log_info "Ollama server stopped"
}

cmd_status() {
    echo "=== Ollama Status ==="
    echo ""
    
    # Check installation
    if is_ollama_installed; then
        echo -e "Installation: ${GREEN}Installed${NC}"
        ollama --version 2>/dev/null | head -1 || true
    else
        echo -e "Installation: ${RED}Not installed${NC}"
        echo "  Run: $0 install"
        return 1
    fi
    echo ""
    
    # Check server
    if is_ollama_running; then
        echo -e "Server:       ${GREEN}Running${NC} (${OLLAMA_BASE_URL})"
    else
        echo -e "Server:       ${RED}Not running${NC}"
        echo "  Run: $0 start"
        return 1
    fi
    echo ""
    
    # Check model
    if is_model_available; then
        echo -e "Model:        ${GREEN}${OLLAMA_MODEL} available${NC}"
    else
        echo -e "Model:        ${YELLOW}${OLLAMA_MODEL} not found${NC}"
        echo "  Run: $0 pull"
    fi
    echo ""
    
    # List all models
    echo "Available models:"
    curl -s "${OLLAMA_BASE_URL}/api/tags" 2>/dev/null | grep -o "\"name\":\"[^\"]*\"" | sed 's/"name":"//g' | sed 's/"//g' | while read -r model; do
        echo "  - $model"
    done
}

cmd_test() {
    log_info "Running test query..."
    
    if ! is_ollama_running; then
        log_error "Ollama server is not running. Run: $0 start"
        exit 1
    fi
    
    if ! is_model_available; then
        log_error "Model ${OLLAMA_MODEL} is not available. Run: $0 pull"
        exit 1
    fi
    
    echo ""
    echo "Query: 'What is 2 + 2? Reply with just the number.'"
    echo "---"
    
    response=$(curl -s "${OLLAMA_BASE_URL}/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${OLLAMA_MODEL}\", \"prompt\": \"What is 2 + 2? Reply with just the number.\", \"stream\": false}" \
        2>/dev/null)
    
    echo "$response" | grep -o '"response":"[^"]*"' | sed 's/"response":"//g' | sed 's/"//g' || echo "$response"
    echo ""
    echo "---"
    log_info "Test completed!"
}

cmd_setup() {
    log_info "Running full Ollama setup..."
    echo ""
    
    cmd_install
    echo ""
    
    cmd_start
    echo ""
    
    cmd_pull
    echo ""
    
    cmd_status
    echo ""
    
    log_info "Setup complete! You can now use Ollama with ai-docs-bot."
    echo ""
    echo "Add to your .env file:"
    echo "  LLM_PROVIDER=ollama"
    echo "  OLLAMA_MODEL=${OLLAMA_MODEL}"
    echo "  OLLAMA_BASE_URL=${OLLAMA_BASE_URL}"
}

# Main
case "${1:-setup}" in
    install)
        cmd_install
        ;;
    pull)
        cmd_pull
        ;;
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    status)
        cmd_status
        ;;
    test)
        cmd_test
        ;;
    setup)
        cmd_setup
        ;;
    *)
        echo "Usage: $0 {install|pull|start|stop|status|test|setup}"
        exit 1
        ;;
esac
