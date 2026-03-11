#!/usr/bin/env bash

set -e

echo "Setting up development environment..."

OS="$(uname -s)"

install_macos() {
    echo "Detected macOS"

    if ! command -v brew >/dev/null 2>&1; then
        echo "Homebrew not found. Install from https://brew.sh"
        exit 1
    fi

    echo "Installing LLVM (clang-format, clang-tidy)..."
    brew install llvm || true

    LLVM_PATH="/opt/homebrew/opt/llvm/bin"

    if ! grep -q "$LLVM_PATH" "$HOME/.zshrc" 2>/dev/null; then
        echo "Adding LLVM to PATH in ~/.zshrc"
        echo "export PATH=\"$LLVM_PATH:\$PATH\"" >> "$HOME/.zshrc"
    fi
}

install_linux() {
    echo "Detected Linux"

    if command -v apt >/dev/null 2>&1; then
        sudo apt update
        sudo apt install -y clang clang-format clang-tidy llvm python3-venv
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y clang clang-tools-extra llvm python3
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Sy --noconfirm clang llvm python
    else
        echo "Unsupported package manager. Install clang-format and clang-tidy manually."
    fi
}

install_windows() {
    echo "Detected Windows"

    if command -v winget >/dev/null 2>&1; then
        winget install -e --id LLVM.LLVM
    else
        echo "Please install LLVM from https://llvm.org/"
    fi
}

case "$OS" in
    Darwin)
        install_macos
        ;;
    Linux)
        install_linux
        ;;
    MINGW*|MSYS*|CYGWIN*)
        install_windows
        ;;
    *)
        echo "Unknown OS: $OS"
        ;;
esac

echo "Setup complete."