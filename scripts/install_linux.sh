#!/bin/bash
set -e

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH."
    exit 1
fi

echo "Setting up BitVoice alias..."

ALIAS_CMD='bitvoice() { docker run --gpus all --rm -v "$(pwd):/workspace" -w /workspace suryaanshrai515/bitvoice:latest "$@"; }'
export -f bitvoice

# Detect shell
SHELL_NAME=$(basename "$SHELL")
RC_FILE=""

if [ "$SHELL_NAME" = "bash" ]; then
    RC_FILE="$HOME/.bashrc"
    [ -f "$HOME/.bash_profile" ] && RC_FILE="$HOME/.bash_profile"
elif [ "$SHELL_NAME" = "zsh" ]; then
    RC_FILE="$HOME/.zshrc"
fi

echo "✅ BitVoice function is available in this session (if sourced)."

if [ -n "$RC_FILE" ]; then
    # Use /dev/tty for input if inside a pipe or script
    read -p "Do you want to add this alias to $RC_FILE? (y/n) " -n 1 -r < /dev/tty || true
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if grep -q "bitvoice()" "$RC_FILE"; then
             echo "Info: Alias already exists in $RC_FILE"
        else
             echo "" >> "$RC_FILE"
             echo "# BitVoice Alias" >> "$RC_FILE"
             echo "$ALIAS_CMD" >> "$RC_FILE"
             echo "✅ Added to $RC_FILE. Please restart your shell or run 'source $RC_FILE'."
        fi
    fi
else
    echo "Warning: Could not detect shell configuration file. Please manually add:"
    echo "$ALIAS_CMD"
fi
