"""
Just-Prompt entry module
"""
import argparse
import sys
from typing import List, Optional


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Just-Prompt - MCP server with unified interface for LLM providers")
    
    parser.add_argument(
        "--default-models",
        type=str,
        default="anthropic:claude-3-7-sonnet-20250219",
        help="Comma-separated list of default models"
    )
    
    # More command line arguments will be added here in the future
    
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Main entry function"""
    parsed_args = parse_args(args)
    
    # Server startup logic will be added here
    print(f"Starting Just-Prompt server, default models: {parsed_args.default_models}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 