#!/usr/bin/env python3
"""
Simple verification script for Happy Coder installation.

This script checks if Happy Coder is properly cloned and describes its functionality.
"""

import os
import json

def verify_happy_installation():
    """Verify that Happy Coder repository is cloned and accessible."""

    happy_dir = "/mnt/disk1/yiliu7/SageAttention-Fork/happy"
    cli_dir = os.path.join(happy_dir, "packages", "happy-cli")
    package_json_path = os.path.join(cli_dir, "package.json")

    print("Happy Coder Installation Verification")
    print("=" * 40)

    # Check if repository is cloned
    if not os.path.exists(happy_dir):
        print("❌ Happy repository not found")
        return False

    print("✅ Happy repository cloned successfully")
    print(f"   Location: {happy_dir}")

    # Check if CLI package exists
    if not os.path.exists(cli_dir):
        print("❌ Happy CLI package not found")
        return False

    print("✅ Happy CLI package found")
    print(f"   Location: {cli_dir}")

    # Check package.json for version info
    if os.path.exists(package_json_path):
        try:
            with open(package_json_path, 'r') as f:
                package_data = json.load(f)

            print(f"✅ Package version: {package_data.get('version', 'unknown')}")
            print(f"   Description: {package_data.get('description', 'N/A')}")

            # Check if binary exists
            bin_path = os.path.join(cli_dir, "bin", "happy.mjs")
            if os.path.exists(bin_path):
                print(f"✅ Binary found: {bin_path}")
            else:
                print(f"⚠️  Binary not found: {bin_path}")

        except json.JSONDecodeError:
            print("⚠️  Could not parse package.json")

    print("\nWhat is Happy Coder?")
    print("-" * 20)
    print("Happy Coder is a mobile and web client for Claude Code & Codex that enables:")
    print("• 📱 Remote control of Claude Code from mobile devices")
    print("• 🔔 Push notifications when Claude needs permissions or encounters errors")
    print("• ⚡ Instant device switching between mobile and desktop")
    print("• 🔐 End-to-end encryption for all communications")
    print("• 🛠️ Open source design with no telemetry or tracking")

    print("\nInstallation Status:")
    print("-" * 18)
    print("✅ Repository successfully cloned from https://github.com/slopus/happy")
    print("⚠️  Dependencies not installed (network connectivity issues)")
    print("⚠️  Project not built (requires Node.js 18+ and yarn)")

    print("\nTo complete installation (when network is available):")
    print("1. Update Node.js to version 18 or higher")
    print("2. Install yarn: npm install -g yarn")
    print("3. Install dependencies: cd happy && yarn install")
    print("4. Install globally: npm install -g happy-coder")
    print("5. Run: happy --help")

    return True

if __name__ == "__main__":
    verify_happy_installation()