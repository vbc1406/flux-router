"""Entry point for `python -m router`.

Flux is a library, not a standalone service.
This file exists so `python -m router` and the Docker default
CMD exit cleanly with a helpful message instead of an error.
"""

import sys


def main() -> int:
    print("Flux router loaded successfully.")
    print("This module is a library — integrate it into your application.")
    print("See README.md for usage.")
    print("To run the demo: python -m router.demo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
