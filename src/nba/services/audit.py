"""Minimal SQLAlchemy/SQLite decision audit store with immutable decision IDs."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import Column, MetaData, String, Table, Text, create_engine, insert, select
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError

class AuditStore:
    def __init__(self, path: Path = Path("data/decisions.sqlite")):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(URL.create("sqlite", database=str(self.path.resolve())))
        metadata = MetaData()
        self.records = Table("decisions", metadata,
            Column("decision_id", String(120), primary_key=True),
            Column("created_at_utc", String(40), nullable=False),
            Column("evidence_json", Text, nullable=False))
        metadata.create_all(self.engine)

    def record(self, decision_id: str, evidence: dict) -> None:
        payload = json.dumps(evidence, sort_keys=True, allow_nan=False)
        try:
            with self.engine.begin() as connection:
                connection.execute(insert(self.records).values(decision_id=decision_id,
                    created_at_utc=datetime.now(timezone.utc).isoformat(), evidence_json=payload))
        except IntegrityError as exc:
            raise ValueError(f"Decision ID already exists and audit records are immutable: {decision_id}") from exc

    def get(self, decision_id: str) -> dict | None:
        statement = select(self.records).where(self.records.c.decision_id == decision_id)
        with self.engine.connect() as connection:
            row = connection.execute(statement).mappings().first()
        if row is None: return None
        return {"decision_id": row["decision_id"], "created_at_utc": row["created_at_utc"],
            "evidence": json.loads(row["evidence_json"])}

    def close(self) -> None:
        self.engine.dispose()
