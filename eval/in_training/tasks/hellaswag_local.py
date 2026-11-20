"""Document processing of the stock lm-eval hellaswag task, taken from the installed lm-eval."""
from lm_eval.tasks.hellaswag.utils import process_docs

# itr_hellaswag.yaml names this function with !function, so it is the interface of this module.
# Listing it here also tells linters that the import is used.
__all__ = ["process_docs"]
