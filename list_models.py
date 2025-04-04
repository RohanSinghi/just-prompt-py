#!/usr/bin/env python3
"""
Tool to list available LLM models

Usage:
    python list_models.py [provider]
    
Arguments:
    provider: Provider to list models for (e.g., 'openai' or 'o')
"""
import argparse
import sys
from typing import List, Optional

from src.just_prompt.atoms.shared.data_types import Provider


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="List available LLM models")
    parser.add_argument(
        "provider",
        nargs="?",
        type=str,
        help="Provider to list models for (e.g., 'openai' or 'o')"
    )
    return parser.parse_args()


def main() -> int:
    """Main function"""
    args = parse_args()
    
    if args.provider:
        try:
            provider = Provider.from_prefix(args.provider)
            print(f"Listing models for {provider.value}:")
            # Logic to get models for the specified provider will be added here
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    else:
        print("Listing models for all providers:")
        # Logic to get models for all providers will be added here
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 