"""Wire contracts and identity. This package deliberately owns no business database."""
from .auth import Actor, require_scope, authenticated_actor
from .client import ServiceClient, ServiceError
from .web import create_app
from .util import canonical_json, digest, utcnow, canonical_region

__version__ = "1.0.0"
__all__ = ["Actor", "require_scope", "authenticated_actor", "ServiceClient",
           "ServiceError", "create_app", "canonical_json", "digest", "utcnow", "canonical_region"]
