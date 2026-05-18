'''Executable module for llm_deploy_prediction.

This module allows the package to be executed directly using `python -m llm_deploy_prediction`.
It delegates to the main CLI entry point.
'''

from llm_deploy_prediction.cli import main

if __name__ == '__main__':
    main()
