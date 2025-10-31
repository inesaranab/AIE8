#!/usr/bin/env python3
"""
Configuration script for guardrails API key.
Uses the official guardrails configure command with proper format.

Usage:
    uv run python configure_guardrails.py [API_KEY]

Or set the API key via environment variable:
    export GUARDRAILS_API_KEY=your_api_key_here
    uv run python configure_guardrails.py
"""

import os
import sys
import subprocess
from pathlib import Path


def get_config_path() -> Path:
    """Get the guardrails config file path."""
    home = Path.home()
    # Guardrails stores config in ~/.guardrails/credentials.json
    config_dir = home / ".guardrails"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "credentials.json"


def configure_guardrails_with_cli(api_key: str):
    """Configure guardrails API key using the official CLI command."""
    try:
        # Use the official guardrails configure command
        # --disable-metrics avoids interactive prompt
        result = subprocess.run(
            [
                sys.executable, "-m", "guardrails", "configure",
                "--token", api_key,
                "--disable-metrics",
                "--disable-remote-inference"
            ],
            input="",  # Empty input for any prompts
            text=True,
            capture_output=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ Guardrails API key configured successfully!")
            print("   Using official guardrails configure command")
            return True
        else:
            print("⚠️  Official command had issues, trying fallback method...")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"⚠️  Official command failed: {e}")
        print("   Trying fallback method...")
        return False


def configure_guardrails_fallback(api_key: str):
    """Fallback: Configure guardrails API key by writing to config file."""
    import json
    
    config_path = get_config_path()
    config = {"api_key": api_key}
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Guardrails API key configured (fallback method)!")
    print(f"   Config saved to: {config_path}")
    print(f"   ⚠️  Keep your API key secret and do not commit it to git!")
    print(f"\n   Note: If guards still fail to install, your token may be expired.")
    print(f"   Get a new token from: https://hub.guardrailsai.com/keys")


def main():
    # Check for API key in environment variable first
    api_key = os.getenv("GUARDRAILS_API_KEY")
    
    # Then check command line argument
    if not api_key and len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    # If still no key, prompt for it
    if not api_key:
        print("🛡️  Guardrails AI Configuration")
        print("   Get your API key from: https://hub.guardrailsai.com/keys")
        print()
        api_key = input("Enter your Guardrails AI API key: ").strip()
    
    if not api_key:
        print("❌ Error: API key is required")
        print("\nUsage:")
        print("  uv run python configure_guardrails.py [API_KEY]")
        print("  or")
        print("  export GUARDRAILS_API_KEY=your_key")
        print("  uv run python configure_guardrails.py")
        sys.exit(1)
    
    # Try official command first, fallback if it fails
    if not configure_guardrails_with_cli(api_key):
        configure_guardrails_fallback(api_key)


if __name__ == "__main__":
    main()

