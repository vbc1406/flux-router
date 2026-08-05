"""Entry point for `python -m router`.

Delegates to the same CLI as the installed `flux` command, so
`python -m router serve` and `flux serve` behave identically — the module
form matters for the Docker default CMD and for source checkouts where the
console script was never installed.
"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
