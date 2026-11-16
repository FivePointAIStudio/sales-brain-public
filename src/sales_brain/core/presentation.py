"""SQLite reference for durable card dedupe; it never calls a channel.

The stable display_key is resolved by the catalogue owner, including approved
same-product aliases. Policy revision, case/session, facts and signed URLs are
NOT identity inputs. Atomic reservation + outbox is local reliability; it does
not promise exactly-once effects from a provider without idempotency/query.

Both verifier callbacks are mandatory trusted host services. A model-provided
`verified: true`, a source ID alone, or a generic product selection is not an
explicit request to show the same picture again.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import sqlite3
import time
from typing import Any, Callable, Mapping
import uuid


PENDING = frozenset({"queued", "submitted", "accepted", "unknown"})
SHOWN = frozenset({"sent", "delivered", "rendered", "read"})
STATES = PENDING | SHOWN | {"confirmed_not_sent"}
MODES = frozenset({"new_card", "explicit_redisplay", "media_only"})


class PresentationError(ValueError):
    pass


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 300:
        raise PresentationError("invalid_" + label)
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class PresentationScope:
    tenant_ref: str
    environment: str
    customer_ref: str
    region: str

    def __post_init__(self):
        for field in ("tenant_ref", "environment", "customer_ref", "region"):
            _nonempty(getattr(self, field), field)

    @property
    def key(self) -> str:
        return _hash(asdict(self))


@dataclass(frozen=True)
class DisplayIntent:
    event_ref: str
    source_revision: str
    display_key: str
    kind: str  # product_search, product_compare, product_reshow, product_media_request
    media_ref: str | None = None


@dataclass(frozen=True)
class DeliveryReceipt:
    source_receipt_ref: str
    scope: PresentationScope
    presentation_id: str
    message_id: str
    state: str
    evidence_kind: str  # provider, client_render, authoritative_query
    attempt_no: int = 1


@dataclass(frozen=True)
class Reservation:
    presentation_id: str
    message_id: str
    mode: str
    status: str
    created: bool
    parent_presentation_id: str | None


EventVerifier = Callable[[PresentationScope, DisplayIntent], bool]
ReceiptVerifier = Callable[[DeliveryReceipt], bool]


class PresentationStore:
    """One SQLite connection per store/thread; other processes may share path."""

    def __init__(self, path: str, *, verify_event: EventVerifier,
                 verify_receipt: ReceiptVerifier, clock: Callable[[], float] = time.time):
        if not callable(verify_event) or not callable(verify_receipt):
            raise PresentationError("trusted_verifiers_required")
        self.verify_event, self.verify_receipt, self.clock = verify_event, verify_receipt, clock
        self.connection = sqlite3.connect(path, timeout=10, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=10000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS presentations (
            presentation_id TEXT PRIMARY KEY,
            scope_key TEXT NOT NULL,
            scope_json TEXT NOT NULL,
            display_key TEXT NOT NULL,
            mode TEXT NOT NULL,
            request_key TEXT NOT NULL,
            event_ref TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            media_ref TEXT,
            case_ref TEXT,
            session_ref TEXT,
            fact_version TEXT,
            parent_presentation_id TEXT REFERENCES presentations(presentation_id),
            message_id TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            attempt_no INTEGER NOT NULL DEFAULT 1,
            payload_digest TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(scope_key, display_key, request_key)
        );
        CREATE TABLE IF NOT EXISTS display_registry (
            scope_key TEXT NOT NULL,
            display_key TEXT NOT NULL,
            presentation_id TEXT NOT NULL REFERENCES presentations(presentation_id),
            PRIMARY KEY(scope_key, display_key)
        );
        CREATE TABLE IF NOT EXISTS presentation_outbox (
            message_id TEXT PRIMARY KEY,
            presentation_id TEXT NOT NULL UNIQUE REFERENCES presentations(presentation_id),
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            retry_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS presentation_receipts (
            source_receipt_ref TEXT PRIMARY KEY,
            presentation_id TEXT NOT NULL REFERENCES presentations(presentation_id),
            receipt_digest TEXT NOT NULL,
            state TEXT NOT NULL,
            applied_state TEXT NOT NULL,
            recorded_at REAL NOT NULL
        );
        """)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    @contextmanager
    def _transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _verify_request(self, scope: PresentationScope, display_key: str,
                        intent: DisplayIntent, mode: str) -> None:
        _nonempty(display_key, "display_key")
        for name in ("event_ref", "source_revision", "display_key", "kind"):
            _nonempty(getattr(intent, name), name)
        if scope.environment == "shadow":
            raise PresentationError("shadow_cannot_reserve_delivery")
        if intent.display_key != display_key or self.verify_event(scope, intent) is not True:
            raise PresentationError("unverified_or_out_of_scope_display_event")
        allowed = {
            "new_card": {"product_search", "product_compare"},
            "explicit_redisplay": {"product_reshow"},
            "media_only": {"product_media_request"},
        }
        if mode not in allowed or intent.kind not in allowed[mode]:
            raise PresentationError("event_does_not_authorize_display_mode")
        if mode == "media_only":
            _nonempty(intent.media_ref, "media_ref")

    @staticmethod
    def _result(row: sqlite3.Row, *, created: bool = False) -> Reservation:
        if created:
            mode = row["mode"]
        elif row["status"] in PENDING:
            mode = "pending_existing"
        elif row["status"] == "confirmed_not_sent":
            mode = "retry_existing"
        else:
            mode = "reference_existing"
        return Reservation(row["presentation_id"], row["message_id"], mode,
                           row["status"], created, row["parent_presentation_id"])

    def _get(self, scope: PresentationScope, presentation_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM presentations WHERE presentation_id=? AND scope_key=?",
            (presentation_id, scope.key),
        ).fetchone()
        if row is None:
            raise PresentationError("presentation_not_found_in_scope")
        return row

    def reserve(self, scope: PresentationScope, *, display_key: str,
                intent: DisplayIntent, mode: str = "new_card",
                stable_payload: Mapping[str, Any] | None = None,
                case_ref: str | None = None, session_ref: str | None = None,
                fact_version: str | None = None) -> Reservation:
        """Reserve a RECOMMEND_PRODUCTS item and its outbox atomically.

        stable_payload comes from a presenter that has validated catalogue facts
        and media ownership, never directly from model JSON. The ledger enforces
        mechanical identity bindings too; it cannot validate ERP facts itself.
        Signed URLs are resolved by the delivery adapter just before send. A
        product_interest event never authorizes redisplay or a new card.
        """
        self._verify_request(scope, display_key, intent, mode)
        payload = dict(stable_payload or {})
        # Prevent accidentally binding a volatile URL to an immutable command.
        def contains_url(value: Any) -> bool:
            if isinstance(value, str):
                return value.startswith(("https://", "http://"))
            if isinstance(value, Mapping):
                return any(contains_url(v) for v in value.values())
            return isinstance(value, (list, tuple)) and any(contains_url(v) for v in value)
        if contains_url(payload):
            raise PresentationError("resolve_media_urls_at_delivery_time")
        if "presentation_id" in payload or "message_id" in payload:
            raise PresentationError("delivery_identity_is_server_owned")
        bound_fields = {**asdict(scope), "display_key": display_key, "mode": mode,
                        "media_ref": intent.media_ref if mode == "media_only" else None,
                        "event_ref": intent.event_ref, "source_revision": intent.source_revision}
        for key, expected in bound_fields.items():
            if key in payload and payload.pop(key) != expected:
                raise PresentationError("stable_payload_identity_conflict:" + key)
        media_refs = payload.pop("media_refs", [intent.media_ref] if mode == "media_only" else [])
        if (not isinstance(media_refs, list) or len(media_refs) > 32
                or any(not isinstance(ref, str) or not ref for ref in media_refs)
                or len(set(media_refs)) != len(media_refs)):
            raise PresentationError("invalid_payload_media_refs")
        if mode == "media_only" and media_refs != [intent.media_ref]:
            raise PresentationError("media_payload_does_not_match_verified_request")
        # The adapter consumes this envelope's authoritative fields, not identity
        # values nested inside content. The presenter still verifies all product
        # facts and the ownership of media_refs before invoking this store.
        command = {**bound_fields, "media_refs": media_refs, "content": payload}
        command_json = _canonical(command)
        if len(command_json.encode()) > 64_000:
            raise PresentationError("stable_payload_too_large")
        command = json.loads(command_json)  # Stable deep snapshot of caller data.
        payload_digest = _hash(command)
        request_key = ("ordinary" if mode == "new_card" else
                       _hash([mode, intent.event_ref, intent.media_ref if mode == "media_only" else None]))
        with self._transaction():
            # Revalidate after obtaining the lock; canceled/stale events cannot
            # reserve a card simply because they passed an earlier read check.
            self._verify_request(scope, display_key, intent, mode)
            existing = self.connection.execute(
                "SELECT * FROM presentations WHERE scope_key=? AND display_key=? AND request_key=?",
                (scope.key, display_key, request_key),
            ).fetchone()
            if existing:
                if mode != "new_card" and existing["payload_digest"] != payload_digest:
                    raise PresentationError("same_explicit_request_changed_command")
                return self._result(existing)
            original = self.connection.execute(
                "SELECT p.* FROM display_registry r JOIN presentations p "
                "ON p.presentation_id=r.presentation_id WHERE r.scope_key=? AND r.display_key=?",
                (scope.key, display_key),
            ).fetchone()
            if mode != "new_card" and original is None:
                raise PresentationError("explicit_reshow_requires_existing_presentation")
            if mode == "new_card" and original is not None:
                return self._result(original)
            if mode == "explicit_redisplay":
                pending = self.connection.execute(
                    "SELECT * FROM presentations WHERE scope_key=? AND display_key=? "
                    "AND mode IN ('new_card','explicit_redisplay') "
                    "AND status IN ('queued','submitted','accepted','unknown') LIMIT 1",
                    (scope.key, display_key),
                ).fetchone()
                if pending:
                    return self._result(pending)
                if original["status"] not in SHOWN:
                    return self._result(original)
            if mode == "media_only":
                pending_media = self.connection.execute(
                    "SELECT * FROM presentations WHERE scope_key=? AND display_key=? "
                    "AND mode='media_only' AND media_ref=? "
                    "AND status IN ('queued','submitted','accepted','unknown') LIMIT 1",
                    (scope.key, display_key, intent.media_ref),
                ).fetchone()
                if pending_media:
                    return self._result(pending_media)
            pid, mid, now = "presentation-" + uuid.uuid4().hex, "message-" + uuid.uuid4().hex, self.clock()
            parent = original["presentation_id"] if original else None
            self.connection.execute(
                "INSERT INTO presentations(presentation_id,scope_key,scope_json,display_key,mode,request_key,"
                "event_ref,source_revision,media_ref,case_ref,session_ref,fact_version,parent_presentation_id,"
                "message_id,status,payload_digest,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (pid, scope.key, _canonical(asdict(scope)), display_key, mode, request_key,
                 intent.event_ref, intent.source_revision, intent.media_ref, case_ref, session_ref,
                 fact_version, parent, mid, "queued", payload_digest, now, now),
            )
            if mode == "new_card":
                self.connection.execute("INSERT INTO display_registry VALUES (?,?,?)", (scope.key, display_key, pid))
            self.connection.execute(
                "INSERT INTO presentation_outbox(message_id,presentation_id,payload_json,status) VALUES (?,?,?,?)",
                (mid, pid, _canonical({**command, "presentation_id": pid, "message_id": mid}), "queued"),
            )
            return self._result(self._get(scope, pid), created=True)

    def record_receipt(self, receipt: DeliveryReceipt) -> Reservation:
        _nonempty(receipt.source_receipt_ref, "source_receipt_ref")
        if receipt.state not in STATES or self.verify_receipt(receipt) is not True:
            raise PresentationError("unverified_delivery_receipt")
        if type(receipt.attempt_no) is not int or receipt.attempt_no < 1:
            raise PresentationError("invalid_receipt_attempt")
        if receipt.state == "confirmed_not_sent" and receipt.evidence_kind != "authoritative_query":
            raise PresentationError("not_sent_requires_authoritative_target_query")
        if receipt.state == "rendered" and receipt.evidence_kind != "client_render":
            raise PresentationError("rendered_requires_scoped_client_ack")
        if receipt.evidence_kind not in {"provider", "client_render", "authoritative_query"}:
            raise PresentationError("unsupported_receipt_source")
        fingerprint = _hash(asdict(receipt))
        with self._transaction():
            row = self._get(receipt.scope, receipt.presentation_id)
            if row["message_id"] != receipt.message_id:
                raise PresentationError("receipt_message_binding_mismatch")
            if receipt.attempt_no > row["attempt_no"]:
                raise PresentationError("receipt_from_unknown_future_attempt")
            previous = self.connection.execute(
                "SELECT * FROM presentation_receipts WHERE source_receipt_ref=?",
                (receipt.source_receipt_ref,),
            ).fetchone()
            if previous:
                if previous["receipt_digest"] != fingerprint:
                    raise PresentationError("receipt_identity_conflict")
                return self._result(row)
            applied = self._next_receipt_state(row, receipt)
            self.connection.execute(
                "INSERT INTO presentation_receipts VALUES (?,?,?,?,?,?)",
                (receipt.source_receipt_ref, receipt.presentation_id, fingerprint,
                 receipt.state, applied, self.clock()),
            )
            self.connection.execute(
                "UPDATE presentations SET status=?,updated_at=? WHERE presentation_id=?",
                (applied, self.clock(), receipt.presentation_id),
            )
            self.connection.execute(
                "UPDATE presentation_outbox SET status=? WHERE presentation_id=?",
                (applied, receipt.presentation_id),
            )
            return self._result(self._get(receipt.scope, receipt.presentation_id))

    @staticmethod
    def _next_receipt_state(row: sqlite3.Row, receipt: DeliveryReceipt) -> str:
        """Monotone progress, with an explicit uncertainty barrier.

        Only reserve/retry can produce a dispatchable queued state after initial
        creation. A delayed queued callback must never re-arm an old command.
        Unknown is resolved by positive exposure evidence or an authoritative
        query; delayed submitted/accepted callbacks are only audit evidence.
        """
        current, incoming = row["status"], receipt.state
        if receipt.attempt_no < row["attempt_no"] and incoming not in SHOWN:
            return current
        shown_rank = {"sent": 1, "delivered": 2, "rendered": 3, "read": 4}
        if current in SHOWN:
            return incoming if incoming in SHOWN and shown_rank[incoming] > shown_rank[current] else current
        if incoming in SHOWN:
            return incoming
        if incoming == "queued":
            return current  # Initial queued/retry is a local atomic operation.
        if incoming == "unknown":
            return "unknown"  # Conservative stop, never a resend permission.
        if incoming == "confirmed_not_sent":
            # record_receipt already requires an authoritative query here.
            return incoming
        if current in {"unknown", "confirmed_not_sent"}:
            return incoming if receipt.evidence_kind == "authoritative_query" else current
        progress_rank = {"queued": 0, "submitted": 1, "accepted": 2}
        return incoming if progress_rank[incoming] > progress_rank[current] else current

    def retry_confirmed_not_sent(self, scope: PresentationScope, *, presentation_id: str,
                                 current_intent: DisplayIntent) -> Reservation:
        """Requeue the same message only after a verified no-effect result.

        The channel adapter must still guarantee idempotency for that message_id
        and query unresolved previous attempts. No timer or status=unknown path
        can call this successfully.
        """
        with self._transaction():
            row = self._get(scope, presentation_id)
            self._verify_request(scope, row["display_key"], current_intent, row["mode"])
            if row["status"] != "confirmed_not_sent":
                raise PresentationError("cannot_retry_without_confirmed_no_effect")
            self.connection.execute(
                "UPDATE presentations SET status='queued',attempt_no=attempt_no+1,updated_at=? WHERE presentation_id=?",
                (self.clock(), presentation_id),
            )
            self.connection.execute(
                "UPDATE presentation_outbox SET status='queued',retry_count=retry_count+1 WHERE presentation_id=?",
                (presentation_id,),
            )
            return self._result(self._get(scope, presentation_id))

    def get(self, scope: PresentationScope, presentation_id: str) -> dict[str, Any]:
        return dict(self._get(scope, presentation_id))

    def get_outbox(self, scope: PresentationScope, presentation_id: str) -> dict[str, Any]:
        """Scoped read for a trusted worker; not an atomic dispatch claim."""
        self._get(scope, presentation_id)
        row = self.connection.execute(
            "SELECT * FROM presentation_outbox WHERE presentation_id=?", (presentation_id,),
        ).fetchone()
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def counts(self) -> dict[str, int]:
        return {name: self.connection.execute("SELECT count(*) FROM " + name).fetchone()[0]
                for name in ("presentations", "display_registry", "presentation_outbox", "presentation_receipts")}
