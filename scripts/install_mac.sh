#!/bin/bash

# macOS install script (similar to Linux but handles Mac specifics if any)
# Currently identical logic to Linux script as Docker Desktop for Mac handles binds similarly.

echo "Setting up BitVoice alias..."

ALIAS_CMD='bitvoice() { docker run --rm -v "$(pwd):/workspace" -w /workspace bitvoice:latest "$@"; }'

# Detect shell
SHELL_NAME=$(basename "$SHELL")
RC_FILE=""

if [ "$SHELL_NAME" = "bash" ]; then
    # Start with .bash_profile for Mac
    RC_FILE="$HOME/.bash_profile"
    [ ! -f "$RC_FILE" ] && RC_FILE="$HOME/.bashrc"
elif [ "$SHELL_NAME" = "zsh" ]; then
    RC_FILE="$HOME/.zshrc"
fi

if [ -n "$RC_FILE" ]; then
    read -p "Do you want to add this alias to $RC_FILE? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if grep -q "bitvoice()" "$RC_FILE"; then
             echo "Alias already exists in $RC_FILE"
        else
             echo "" >> "$RC_FILE"
             echo "# BitVoice Alias" >> "$RC_FILE"
             echo "$ALIAS_CMD" >> "$RC_FILE"
             echo "Added to $RC_FILE. Please restart your shell or run 'source $RC_FILE'."
        fi
    fi
else
    echo "Could not detect shell configuration file. Please manually add:"
    echo "$ALIAS_CMD"
fi
