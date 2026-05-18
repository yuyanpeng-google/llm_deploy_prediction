'''Roofline model for calculating theoretical performance of LLM deployment.

This script is a wrapper around the refactored roofline modules.
It delegates execution to the `cli` module.
'''

import sys
from llm_deploy_prediction.roofline.cli import main

if __name__ == '__main__':
    main(sys.argv[1:])
