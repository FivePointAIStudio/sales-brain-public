"""Read-only, evidence-bound product identity and conversation helpers.

The Agent selects a product reference from the whole message. This module does
not classify customer language, create business actions, send messages, query
prices, or decide that a request to contact sales means a callback request.

Candidates are supplied by authenticated context/catalog adapters, never copied
from model arguments. Context candidates must be scoped to the customer. A
catalog callback receives ``(reference, region)`` and returns ``status``,
``candidates``, ``complete`` and ``source_refs``. A partial catalog page cannot
establish a unique bare-number identity. Resolved identity does not prove live
availability, price eligibility, an order, or any other completed business act.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Any
import unicodedata


CatalogLookup = Callable[[str, str], Mapping[str, Any]]
_CODE = re.compile(r"[A-Z0-9][A-Z0-9_-]{0,63}\Z")
_NUMBER = re.compile(r"(?:[A-Z]+[-_]?)?([0-9]+)\Z")
CONFIRMED_DISPLAY_STATES = frozenset({"rendered", "sent", "delivered", "read"})
_UNCONFIRMED = frozenset({
    "preview_generated", "generated", "planned", "pending", "queued", "failed",
    "unknown", "cancelled", "superseded", "removed", "inactive", "revoked",
})


def normalize_product_reference(reference: str) -> str | None:
    """Normalize a reference field, not an entire customer utterance.

    Leading zeros and explicit prefixes are preserved. NFKC handles full-width
    Latin characters; whitespace between code characters is insignificant.
    Units, prices, dates and free-form sentences are not product references.
    """
    if not isinstance(reference, str) or len(reference) > 128:
        return None
    value = "".join(unicodedata.normalize("NFKC", reference).upper().split())
    # Preserve the legacy catalog's public-code alphabet, including suffixes
    # such as SYN-10001-1. All-numeric references have already been selected as
    # product references by the Agent, so this is not a quantity classifier.
    if not _CODE.fullmatch(value) or (not any(c.isalpha() for c in value) and not value.isdigit()):
        return None
    return value

