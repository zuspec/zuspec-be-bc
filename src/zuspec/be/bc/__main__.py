"""``python -m zuspec.be.bc``: the ``zuspec-bc`` command line (:mod:`.cli`)."""
import sys

from .cli import main

sys.exit(main())
