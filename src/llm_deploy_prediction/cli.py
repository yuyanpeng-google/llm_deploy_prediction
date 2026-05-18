'''Main CLI entry point for llm_deploy_prediction.

This module provides a central CLI with sub-commands for different tools
in the package.
'''

import sys
from typing import List, Optional

from llm_deploy_prediction.roofline.cli import main as roofline_main


def main(argv: Optional[List[str]] = None) -> None:
    '''Main entry point for the CLI.

    Dispatches to the appropriate module based on the first argument.

    Args:
        argv: List of arguments to parse. If None, uses sys.argv.
    '''
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: llm_deploy_prediction <command> [options]")
        print("Available commands:")
        print("  roofline - Run roofline analysis")
        sys.exit(1)

    command = argv[0]
    remaining = argv[1:]

    if command == 'roofline':
        roofline_main(remaining)
    else:
        print(f"Unknown command: {command}")
        print("Usage: llm_deploy_prediction <command> [options]")
        print("Available commands:")
        print("  roofline - Run roofline analysis")
        sys.exit(1)


if __name__ == '__main__':
    main()
