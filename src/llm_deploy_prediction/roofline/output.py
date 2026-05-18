'''Output formatting for roofline analysis.

This module contains functions to format and print the results of the
roofline analysis, such as printing markdown tables.
'''

from typing import Any, List


def print_markdown_table(headers: List[str], rows: List[List[Any]]) -> None:
    '''Prints a list of rows as a markdown table.

    Args:
        headers: List of column headers.
        rows: List of rows, where each row is a list of values.
    '''
    if not headers or not rows:
        return

    # Calculate max width for each column
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    # Print header
    header_str = " | ".join(
        f"{h:<{widths[i]}}" for i, h in enumerate(headers)
    )
    print(f"| {header_str} |")

    # Print separator
    sep_str = " | ".join("-" * widths[i] for i in range(len(headers)))
    print(f"| {sep_str} |")

    # Print rows
    for row in rows:
        row_str = " | ".join(
            f"{str(val):<{widths[i]}}" for i, val in enumerate(row)
        )
        print(f"| {row_str} |")
