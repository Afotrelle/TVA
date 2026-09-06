"""Attack helpers for the OpenTAD/TVA pipeline.

This repo snapshot does not ship a full, unified attack registry. Keep the
package importable for the experiment scripts and expose the sparse-PGD entry
point used in the custom comparison runs.
"""

from .sparse_pgd import attack_PGD

__all__ = ["attack_PGD"]
