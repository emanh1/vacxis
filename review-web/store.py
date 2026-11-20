"""Persist assignments, user preferences, and verified revisions."""

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3

from content import changed_content, review_content, review_data
from dataset_format import as_golden, context_title, is_golden
from sampling import allocate_scientific_batches, compute_fleiss_kappa, summarize_allocation

BASE_DIR = Path(__file__).resolve().parent
SCOPES = {"live": "Duyệt thật", "test": "Thử nghiệm nội bộ"}
CONSENT_VERSION = 3
EXPORT_CONSENT_VERSION = 2
QUESTION_ID = re.compile(r"[A-Za-z0-9_-]{1,100}")


class ReviewError(ValueError):
    pass


class ReviewConflict(ReviewError):
    pass


def timestamp(when=None):
    return (when or datetime.now(UTC)).isoformat(timespec="seconds")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def database_location():
    url = os.environ.get("DATABASE_URL")
    if url:
        if not url.startswith(("postgresql://", "postgres://")):
            raise RuntimeError("DATABASE_URL must be a PostgreSQL connection string.")
        return url
    if os.environ.get("VERCEL") or os.environ.get("REVIEW_PUBLIC_URL", "").startswith("https://"):
        raise RuntimeError("Set DATABASE_URL for the hosted app.")
    return os.environ.get("REVIEW_DB_PATH", str(BASE_DIR / "review.sqlite3"))


class Connection:
    def __init__(self, raw, postgres):
        self.raw = raw
        self.postgres = postgres

    def execute(self, sql, params=()):
        return self.raw.execute(sql.replace("?", "%s") if self.postgres else sql, params)

    def one(self, sql, params=()):
        row = self.execute(sql, params).fetchone()
        return dict(row) if row is not None else None

    def all(self, sql, params=()):
        return [dict(row) for row in self.execute(sql, params)]

    def locked(self, sql, params=()):
        return self.one(sql + (" FOR UPDATE" if self.postgres else ""), params)


@contextmanager
def connect(location, *, write=False):
    postgres = location.startswith(("postgresql://", "postgres://"))
    if postgres:
        import psycopg
        from psycopg.rows import dict_row

        raw = psycopg.connect(location, row_factory=dict_row, connect_timeout=8)
    else:
        raw = sqlite3.connect(location, timeout=8)
        raw.row_factory = sqlite3.Row
        raw.execute("PRAGMA foreign_keys = ON")
    try:
        if write and not postgres:
            raw.execute("BEGIN IMMEDIATE")
        yield Connection(raw, postgres)
        raw.commit()
    except BaseException:
        raw.rollback()
        raise
    finally:
        raw.close()


def ensure_schema(db):
    if db.postgres:
        db.execute("SELECT pg_advisory_xact_lock(73418302)")
    for statement in (BASE_DIR / "schema.sql").read_text(encoding="utf-8").split(";"):
        if statement.strip():
            db.execute(statement)
    for table in ("review_items", "expert_reviews"):
        if db.postgres:
            db.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS review_data TEXT NOT NULL DEFAULT '{{}}'")
        elif "review_data" not in {row["name"] for row in db.all(f"PRAGMA table_info({table})")}:
            db.execute(f"ALTER TABLE {table} ADD COLUMN review_data TEXT NOT NULL DEFAULT '{{}}'")


def import_documents(location, path):
    records = json.loads(path.read_text(encoding="utf-8"))
    required = {"chunk_id", "doc_id", "doc_title", "category", "header", "text", "position"}
    if not isinstance(records, list) or not records:
        raise ReviewError("Danh sách đoạn tài liệu phải có nội dung.")
    identifiers = set()
    for row in records:
        if (not isinstance(row, dict) or not required <= row.keys()
                or any(not isinstance(row[key], str) for key in required - {"position"})
                or not isinstance(row["position"], int) or not row["chunk_id"] or not row["doc_id"]
                or row["chunk_id"] in identifiers):
            raise ReviewError("Định dạng đoạn tài liệu không hợp lệ.")
        identifiers.add(row["chunk_id"])
    with connect(location, write=True) as db:
        ensure_schema(db)
        db.locked("SELECT source_hash FROM dataset WHERE id = 1")
        for doc_id in {row["doc_id"] for row in records}:
            db.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))
        for row in records:
            db.execute(
                "INSERT INTO document_chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                tuple(row[key] for key in ("chunk_id", "doc_id", "doc_title", "category", "header", "text", "position")),
            )
    return len(records)


def ensure_allocations(db, *, refresh=False):
    if not refresh and db.one("SELECT 1 FROM question_allocations LIMIT 1"):
        return db.all("SELECT * FROM question_allocations")
    questions = db.all("SELECT id, original FROM questions ORDER BY position")
    if not questions:
        return []
    allocations = allocate_scientific_batches(questions)
    if refresh:
        db.execute("DELETE FROM question_allocations")
    for row in allocations:
        db.execute(
            "INSERT INTO question_allocations (question_id, batch_type, stratum, primary_slot, auditor_slot, requires_audit) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (question_id) DO NOTHING",
            (row["question_id"], row["batch_type"], row["stratum"], row["primary_slot"], row["auditor_slot"], row["requires_audit"]),
        )
    return allocations


def upgrade_eval_import(db, records, source_hash):
    existing = db.all("SELECT id, original FROM questions ORDER BY position")
    if len(existing) != len(records):
        raise ReviewError("Số câu hiện có khác bộ DeepEval được chọn.")
    for current, (identifier, _, record) in zip(existing, records, strict=True):
        golden = as_golden(json.loads(current["original"]))
        if golden is None or current["id"] != identifier or any(
            golden[field] != record[field] for field in ("input", "expected_output", "context")
        ) or any(
            golden["additional_metadata"].get(field) != record["additional_metadata"].get(field)
            for field in ("query_id", "source_chunk_ids", "doc_ids", "answerability", "complexity",
                          "multi_hop_type", "query_style")
        ):
            raise ReviewError(f"Câu {current['id']} không khớp nguồn DeepEval, nên không thể nâng định dạng.")
    for identifier, _, record in records:
        db.execute("UPDATE questions SET original = ? WHERE id = ?", (json.dumps(record, ensure_ascii=False), identifier))
    db.execute("UPDATE dataset SET source_hash = ? WHERE id = 1", (source_hash,))


def import_dataset(location, path, *, upgrade_eval=False):
    contents = path.read_bytes()
    source_hash = hashlib.sha256(contents).hexdigest()
    payload = json.loads(contents.decode("utf-8-sig"))
    native = isinstance(payload, dict) and isinstance(payload.get("metadata"), dict)
    records = payload.get("goldens") if native else payload
    if not isinstance(records, list) or not records:
        raise ReviewError("Bộ dữ liệu phải là một danh sách câu hỏi.")
    ids = set()
    entries = []
    for record in records:
        if not isinstance(record, dict):
            raise ReviewError("Mỗi câu hỏi phải là một bản ghi JSON.")
        if native:
            if not is_golden(record) or not isinstance(record["additional_metadata"], dict):
                raise ReviewError("Mỗi golden cần có input, expected_output, context và additional_metadata.")
            metadata = record["additional_metadata"]
            identifier = metadata.get("query_id", "")
            query, answer = record["input"], record["expected_output"]
            if metadata.get("answerability") not in ("ANSWERABLE", "PARTIALLY_ANSWERABLE", "UNANSWERABLE"):
                raise ReviewError(f"{identifier}: answerability không hợp lệ.")
            contexts = record["context"]
            chunk_ids = metadata.get("source_chunk_ids")
            if (not isinstance(contexts, list) or not isinstance(chunk_ids, list) or len(contexts) != len(chunk_ids)
                    or any(not isinstance(value, str) or not value for value in (*contexts, *chunk_ids))):
                raise ReviewError(f"{identifier}: context và source_chunk_ids phải khớp từng mục.")
            doc_titles = metadata.get("doc_titles")
            doc_ids = metadata.get("doc_ids")
            context_titles = {title for text in contexts if (title := context_title(text))}
            if (not isinstance(doc_titles, list) or not isinstance(doc_ids, list)
                    or any(title not in doc_titles for title in context_titles)
                    or any(chunk_id.rsplit("_c_", 1)[0] not in doc_ids for chunk_id in chunk_ids)):
                raise ReviewError(f"{identifier}: nguồn không khớp. Hãy chạy prepare_review_dataset.py trước khi nhập.")
        else:
            identifier = record.get("QUERY_ID", "")
            query, answer = record.get("QUERY"), record.get("REFERENCE_ANSWER")
        if not isinstance(identifier, str) or not QUESTION_ID.fullmatch(identifier) or identifier in ids:
            raise ReviewError("QUERY_ID phải hợp lệ và duy nhất.")
        if not isinstance(query, str) or not isinstance(answer, str):
            raise ReviewError(f"{identifier} thiếu nội dung câu hỏi hoặc câu trả lời.")
        ids.add(identifier)
        entries.append((identifier, query, record))

    with connect(location, write=True) as db:
        ensure_schema(db)
        db.execute("INSERT INTO dataset VALUES (1, ?) ON CONFLICT (id) DO NOTHING", (source_hash,))
        existing = db.locked("SELECT source_hash FROM dataset WHERE id = 1")
        if existing["source_hash"] != source_hash:
            if not (native and upgrade_eval):
                raise ReviewError("Cơ sở dữ liệu đã chứa một phiên bản khác. Hãy dùng cơ sở dữ liệu mới.")
            upgrade_eval_import(db, entries, source_hash)
        if native:
            db.execute(
                "INSERT INTO dataset_metadata (id, metadata) VALUES (1, ?) "
                "ON CONFLICT (id) DO UPDATE SET metadata = excluded.metadata",
                (json.dumps(payload["metadata"], ensure_ascii=False),),
            )
        for position, (identifier, query, record) in enumerate(entries):
            db.execute(
                "INSERT INTO questions VALUES (?, ?, ?, ?) ON CONFLICT (id) DO NOTHING",
                (identifier, position, query, json.dumps(record, ensure_ascii=False)),
            )
            for scope in SCOPES:
                db.execute(
                    "INSERT INTO review_items (scope, question_id) VALUES (?, ?) "
                    "ON CONFLICT (scope, question_id) DO NOTHING",
                    (scope, identifier),
                )
    return len(records)


def issue_link(db, reviewer_id, referral_id=None, *, token=None):
    token = token or secrets.token_urlsafe(32)
    db.execute(
        "UPDATE invitations SET consumed_at = ? WHERE reviewer_id = ? AND consumed_at IS NULL",
        (timestamp(), reviewer_id),
    )
    inserted = db.execute(
        "INSERT INTO invitations (token_hash, reviewer_id, referral_id, expires_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT (token_hash) DO NOTHING",
        (digest(token), reviewer_id, referral_id, timestamp(datetime.now(UTC) + timedelta(days=7))),
    ).rowcount
    if inserted != 1:
        raise ReviewError("Hãy thử tạo lại lời mời.")
    return token


def reserve_operation(db, operation):
    if not isinstance(operation, str) or not re.fullmatch(r"[a-f0-9]{32}", operation):
        raise ReviewError("Hãy tải lại trang quản lý rồi thử lại.")
    changed = db.execute("INSERT INTO admin_operations VALUES (?) ON CONFLICT (id) DO NOTHING", (operation,)).rowcount
    if changed != 1:
        raise ReviewConflict("Thao tác này đã được xử lý. Hãy tải lại trang quản lý.")


def release_unstarted(db, reviewer_id):
    if db.one("SELECT 1 FROM expert_slots WHERE reviewer_id = ? LIMIT 1", (reviewer_id,)):
        raise ReviewError("Không thể thu hồi lẻ câu hỏi của chuyên gia trong chế độ phân chia khoa học.")
    changed = db.execute(
        "UPDATE review_items SET reviewer_id = NULL WHERE reviewer_id = ? AND status = 'pending'",
        (reviewer_id,),
    ).rowcount
    if not changed:
        raise ReviewError("Người duyệt này không còn câu chưa bắt đầu.")
    if not db.one("SELECT question_id FROM review_items WHERE reviewer_id = ? LIMIT 1", (reviewer_id,)):
        db.execute("UPDATE reviewers SET access_version = access_version + 1 WHERE id = ?", (reviewer_id,))
        person = db.one("SELECT * FROM reviewers WHERE id = ?", (reviewer_id,))
        if person["scope"] == "live" and person["consent"] == "accepted":
            update_profile(db, person, "withdrawn", {})
        db.execute("UPDATE invitations SET consumed_at = ? WHERE reviewer_id = ? AND consumed_at IS NULL",
                   (timestamp(), reviewer_id))
    return changed


def invite(db, scope, count, can_refer, referral_id, *, token=None):
    if scope not in SCOPES or not 1 <= count <= 460:
        raise ReviewError("Chọn lượt duyệt và số câu từ 1 đến 460.")
    db.locked("SELECT source_hash FROM dataset WHERE id = 1")
    if db.one("SELECT 1 FROM expert_slots WHERE scope = ? LIMIT 1", (scope,)):
        raise ReviewError("Lượt duyệt này đang chạy chế độ 3 chuyên gia.")
    if referral_id:
        changed = db.execute(
            "UPDATE referrals SET handled_at = ? WHERE id = ? AND handled_at IS NULL",
            (timestamp(), referral_id),
        ).rowcount
        if scope != "live" or changed != 1:
            raise ReviewError("Đề xuất này không còn chờ duyệt.")
    available = db.all(
        "SELECT r.question_id FROM review_items r JOIN questions q ON r.question_id = q.id "
        "WHERE r.scope = ? AND r.reviewer_id IS NULL ORDER BY q.position LIMIT ?",
        (scope, count),
    )
    if len(available) != count:
        raise ReviewError("Lượt duyệt này còn ít câu chưa phân công hơn số đã chọn.")
    reviewer_id = "R-" + secrets.token_hex(8).upper()
    db.execute(
        "INSERT INTO reviewers (id, scope, can_refer) VALUES (?, ?, ?)",
        (reviewer_id, scope, int(can_refer and scope == "live")),
    )
    for row in available:
        db.execute(
            "UPDATE review_items SET reviewer_id = ? WHERE scope = ? AND question_id = ?",
            (reviewer_id, scope, row["question_id"]),
        )
    return issue_link(db, reviewer_id, referral_id, token=token)


def invited_reviewer(db, token):
    return db.one(
        "SELECT r.* FROM invitations i JOIN reviewers r ON r.id = i.reviewer_id "
        "WHERE i.token_hash = ? AND i.consumed_at IS NULL AND i.expires_at > ?",
        (digest(token), timestamp()),
    )


def accept_invite(db, token):
    token_hash = digest(token)
    invitation = db.one("SELECT reviewer_id FROM invitations WHERE token_hash = ?", (token_hash,))
    if not invitation:
        return None
    reviewer_id = invitation["reviewer_id"]
    db.locked("SELECT id FROM reviewers WHERE id = ?", (reviewer_id,))
    now = timestamp()
    changed = db.execute(
        "UPDATE invitations SET consumed_at = ? WHERE token_hash = ? "
        "AND consumed_at IS NULL AND expires_at > ?",
        (now, token_hash, now),
    ).rowcount
    if changed != 1:
        return None
    db.execute("UPDATE reviewers SET access_version = access_version + 1 WHERE id = ?", (reviewer_id,))
    return db.one("SELECT * FROM reviewers WHERE id = ?", (reviewer_id,))


def update_profile(db, person, decision, profile):
    now = timestamp()
    changed = db.execute(
        "UPDATE reviewers SET consent = ?, consent_at = ?, consent_version = ?, full_name = ?, "
        "profession = ?, workplace = ?, profile_revision = profile_revision + 1 "
        "WHERE id = ? AND scope = 'live' AND profile_revision = ?",
        (decision, now, CONSENT_VERSION, profile.get("full_name"), profile.get("profession"),
         profile.get("workplace"), person["id"], person["profile_revision"]),
    ).rowcount
    if changed != 1:
        raise ReviewConflict("Lựa chọn chia sẻ thông tin vừa thay đổi. Hãy tải lại trang.")


def record_event(db, item, actor_id, now):
    snapshot = {field: item[field] for field in ("answer", "answerability", "evidence_ids", "notes", "verified_at")}
    snapshot["review_data"] = json.loads(item.get("review_data") or "{}")
    db.execute(
        "INSERT INTO review_events VALUES (?, ?, ?, ?, ?, ?, ?)",
        (item["scope"], item["question_id"], item["revision"], actor_id, item["status"], now,
         json.dumps(snapshot, ensure_ascii=False)),
    )


def save_review(db, item, reviewer_id, revision, status, answer, answerability, evidence_ids, notes, *, review_data=None):
    now = timestamp()
    if item["status"] == "verified" or item["revision"] != revision:
        raise ReviewConflict("Câu này vừa được cập nhật. Hãy mở lại để xem bản mới.")
    updated = dict(item, status=status, revision=revision + 1, answer=answer, answerability=answerability,
                   evidence_ids=json.dumps(evidence_ids), notes=notes, verified_at=now if status == "verified" else None,
                   review_data=json.dumps(review_data or {}, ensure_ascii=False))
    changed = db.execute(
        "UPDATE review_items SET status = ?, revision = ?, answer = ?, answerability = ?, "
        "evidence_ids = ?, notes = ?, verified_at = ?, review_data = ? "
        "WHERE scope = ? AND question_id = ? AND revision = ? AND reviewer_id = ?",
        (status, updated["revision"], answer, answerability, updated["evidence_ids"], notes, updated["verified_at"],
         updated["review_data"], item["scope"], item["question_id"], revision, reviewer_id),
    ).rowcount
    if changed != 1:
        raise ReviewConflict("Câu này vừa được cập nhật. Hãy mở lại để xem bản mới.")
    record_event(db, updated, reviewer_id, now)


def _advance_item(db, item, status, actor_id, now, *, answer=None, answerability=None, evidence_ids=None, notes=None,
                  verified_at=None, review_data=None):
    """Advance review_items revision atomically and append a matching review_events snapshot."""
    next_rev = item["revision"] + 1
    new_answer = item["answer"] if answer is None else answer
    new_label = item["answerability"] if answerability is None else answerability
    new_ev = item["evidence_ids"] if evidence_ids is None else (
        evidence_ids if isinstance(evidence_ids, str) else json.dumps(evidence_ids)
    )
    new_notes = item["notes"] if notes is None else notes
    new_data = item.get("review_data", "{}") if review_data is None else json.dumps(review_data, ensure_ascii=False)
    db.execute(
        "UPDATE review_items SET status = ?, revision = ?, answer = ?, answerability = ?, "
        "evidence_ids = ?, notes = ?, verified_at = ?, review_data = ? WHERE scope = ? AND question_id = ?",
        (status, next_rev, new_answer, new_label, new_ev, new_notes, verified_at, new_data, item["scope"], item["question_id"]),
    )
    updated = dict(
        item,
        status=status,
        revision=next_rev,
        answer=new_answer,
        answerability=new_label,
        evidence_ids=new_ev,
        notes=new_notes,
        verified_at=verified_at,
        review_data=new_data,
    )
    record_event(db, updated, actor_id, now)
    return updated


def reopen(db, scope, question_id, revision, *, unassign=False):
    item = db.locked("SELECT * FROM review_items WHERE scope = ? AND question_id = ?", (scope, question_id))
    statuses = {"draft", "flagged", "verified"} if unassign else {"flagged", "verified"}
    if not item or item["status"] not in statuses or item["revision"] != revision:
        raise ReviewConflict("Trạng thái câu hỏi đã thay đổi. Hãy tải lại trang.")
    scientific = bool(db.one("SELECT 1 FROM expert_slots WHERE scope = ? LIMIT 1", (scope,)))
    if unassign and scientific:
        raise ReviewError("Không thể trả về danh sách chung khi đang chạy chế độ 3 chuyên gia.")
    updated = dict(item, status="draft", revision=revision + 1, verified_at=None)
    if unassign:
        updated.update(status="pending", answer=None, answerability=None, evidence_ids=None, notes="", review_data="{}")
        db.execute(
            "UPDATE review_items SET reviewer_id = NULL, status = 'pending', revision = ?, verified_at = NULL, "
            "answer = NULL, answerability = NULL, evidence_ids = NULL, notes = '', review_data = '{}' "
            "WHERE scope = ? AND question_id = ?",
            (updated["revision"], scope, question_id),
        )
    else:
        db.execute(
            "UPDATE review_items SET status = 'draft', revision = ?, verified_at = NULL "
            "WHERE scope = ? AND question_id = ?",
            (updated["revision"], scope, question_id),
        )
        if scientific:
            alloc = db.one("SELECT batch_type, requires_audit FROM question_allocations WHERE question_id = ?", (question_id,))
            if alloc and alloc["batch_type"] == "independent" and not alloc["requires_audit"]:
                db.execute(
                    "UPDATE expert_reviews SET status = 'draft', revision = revision + 1 "
                    "WHERE scope = ? AND question_id = ? AND role = 'primary'",
                    (scope, question_id),
                )
    record_event(db, updated, "admin", timestamp())


def expert_slot_for(db, reviewer_id):
    return db.one("SELECT scope, slot, reviewer_id FROM expert_slots WHERE reviewer_id = ?", (reviewer_id,))


def setup_scientific_review(db, scope, tokens_by_slot):
    """Create 3 expert slots and distribute calibration, overlap, primary, and cross-audit tasks."""
    if scope not in SCOPES:
        raise ReviewError("Lượt duyệt không hợp lệ.")
    ensure_schema(db)
    db.locked("SELECT source_hash FROM dataset WHERE id = 1")
    if db.one("SELECT 1 FROM expert_slots WHERE scope = ? LIMIT 1", (scope,)):
        raise ReviewError("Lượt duyệt này đã khởi tạo gói cho 3 chuyên gia.")
    if db.one(
        "SELECT 1 FROM review_items WHERE scope = ? AND (reviewer_id IS NOT NULL OR status <> 'pending') LIMIT 1",
        (scope,),
    ):
        raise ReviewError("Lượt duyệt này đã có câu được phân công hoặc duyệt thủ công.")
    allocations = ensure_allocations(db)
    if not allocations:
        raise ReviewError("Chưa có câu hỏi trong cơ sở dữ liệu.")

    slot_rids = {}
    for slot in (1, 2, 3):
        reviewer_id = "R-" + secrets.token_hex(8).upper()
        db.execute("INSERT INTO reviewers (id, scope, can_refer) VALUES (?, ?, 0)", (reviewer_id, scope))
        db.execute("INSERT INTO expert_slots (scope, slot, reviewer_id) VALUES (?, ?, ?)", (scope, slot, reviewer_id))
        issue_link(db, reviewer_id, None, token=tokens_by_slot[slot])
        slot_rids[slot] = reviewer_id

    for row in allocations:
        qid = row["question_id"]
        batch = row["batch_type"]
        if batch in ("calibration", "overlap"):
            for slot in (1, 2, 3):
                db.execute(
                    "INSERT INTO expert_reviews (scope, question_id, reviewer_id, role) VALUES (?, ?, ?, ?)",
                    (scope, qid, slot_rids[slot], batch),
                )
            db.execute(
                "UPDATE review_items SET reviewer_id = ? WHERE scope = ? AND question_id = ?",
                (slot_rids[1], scope, qid),
            )
        else:
            p_rid = slot_rids[row["primary_slot"]]
            db.execute(
                "INSERT INTO expert_reviews (scope, question_id, reviewer_id, role) VALUES (?, ?, ?, 'primary')",
                (scope, qid, p_rid),
            )
            db.execute(
                "UPDATE review_items SET reviewer_id = ? WHERE scope = ? AND question_id = ?",
                (p_rid, scope, qid),
            )
            if row["requires_audit"]:
                a_rid = slot_rids[row["auditor_slot"]]
                db.execute(
                    "INSERT INTO expert_reviews (scope, question_id, reviewer_id, role) VALUES (?, ?, ?, 'auditor')",
                    (scope, qid, a_rid),
                )
    return [{"slot": slot, "reviewer_id": slot_rids[slot]} for slot in (1, 2, 3)]


def reset_scientific_review(db, scope, expected_reviewers):
    """Withdraw an unused scientific assignment and invalidate its invitations."""
    if scope not in SCOPES:
        raise ReviewError("Lượt duyệt không hợp lệ.")
    db.locked("SELECT source_hash FROM dataset WHERE id = 1")
    slots = db.all("SELECT reviewer_id FROM expert_slots WHERE scope = ? ORDER BY reviewer_id", (scope,))
    if not slots:
        raise ReviewError("Lượt duyệt này chưa có phân công 3 chuyên gia.")
    if [slot["reviewer_id"] for slot in slots] != expected_reviewers:
        raise ReviewConflict("Phân công đã thay đổi. Hãy tải lại trang rồi thử lại.")
    for slot in slots:
        db.locked("SELECT id FROM reviewers WHERE id = ?", (slot["reviewer_id"],))
    db.all("SELECT question_id FROM review_items WHERE scope = ? ORDER BY question_id" +
           (" FOR UPDATE" if db.postgres else ""), (scope,))
    started = db.one(
        "SELECT question_id FROM expert_reviews WHERE scope = ? AND "
        "(status <> 'pending' OR revision <> 0 OR answer IS NOT NULL OR answerability IS NOT NULL "
        "OR evidence_ids IS NOT NULL OR notes <> '' OR review_note <> '' OR review_data <> '{}') LIMIT 1",
        (scope,),
    )
    shared_progress = db.one(
        "SELECT question_id FROM review_items WHERE scope = ? AND "
        "(status <> 'pending' OR answer IS NOT NULL OR answerability IS NOT NULL OR evidence_ids IS NOT NULL "
        "OR notes <> '' OR verified_at IS NOT NULL OR review_data <> '{}') LIMIT 1",
        (scope,),
    )
    if started or shared_progress:
        raise ReviewError("Đã có nội dung duyệt được lưu. Hãy liên hệ nhóm để xử lý kết quả trước khi thu hồi toàn bộ.")
    db.execute("DELETE FROM expert_reviews WHERE scope = ?", (scope,))
    db.execute("DELETE FROM expert_slots WHERE scope = ?", (scope,))
    db.execute("UPDATE review_items SET reviewer_id = NULL, revision = revision + 1 WHERE scope = ?", (scope,))
    for slot in slots:
        reviewer_id = slot["reviewer_id"]
        db.execute("DELETE FROM invitations WHERE reviewer_id = ?", (reviewer_id,))
        db.execute("DELETE FROM reviewers WHERE id = ? AND scope = ?", (reviewer_id, scope))
    return len(slots)


def _consensus_check(rows, original):
    """Compare submitted answers, questions, evidence, and final decisions."""
    signatures = set()
    for row in rows:
        ev = tuple(sorted(json.loads(row["evidence_ids"] or "[]")))
        ans = " ".join((row["answer"] or "").split())
        data = review_data(row)
        query = data.get("query") or original.get("input", original.get("QUERY", ""))
        result = data.get("status") or (
            "EDITED" if changed_content(original, query, row["answer"], row["answerability"], ev) else "APPROVED"
        )
        signatures.add((row["answerability"], ev, ans, query, result))
    return len(signatures) == 1


def _combined_notes(rows):
    parts = [row["notes"].strip() for row in rows if row["notes"].strip()]
    return "\n\n".join(dict.fromkeys(parts))


def save_expert_review(db, item, erow, allocation, reviewer_id, revision, status, answer, answerability, evidence_ids, notes,
                       *, review_data=None):
    """Persist an expert's calibration, overlap, or primary submission and advance shared state."""
    now = timestamp()
    if erow["role"] == "auditor":
        raise ReviewError("Vui lòng dùng thao tác kiểm định chéo cho câu hỏi này.")
    if erow["revision"] != revision or erow["status"] in ("submitted", "approved"):
        raise ReviewConflict("Câu này vừa được cập nhật. Hãy mở lại để xem bản mới.")

    new_rev = revision + 1
    ev_json = json.dumps(evidence_ids)
    changed = db.execute(
        "UPDATE expert_reviews SET status = ?, revision = ?, answer = ?, answerability = ?, "
        "evidence_ids = ?, notes = ?, updated_at = ?, review_data = ? "
        "WHERE scope = ? AND question_id = ? AND role = ? AND reviewer_id = ? AND revision = ?",
        (status, new_rev, answer, answerability, ev_json, notes, now, json.dumps(review_data or {}, ensure_ascii=False),
         erow["scope"], erow["question_id"], erow["role"], reviewer_id, revision),
    ).rowcount
    if changed != 1:
        raise ReviewConflict("Câu này vừa được cập nhật. Hãy mở lại để xem bản mới.")

    batch = allocation["batch_type"]
    scope, qid = item["scope"], item["question_id"]
    if batch in ("calibration", "overlap"):
        next_status = item["status"] if item["status"] == "verified" else "draft"
        item = _advance_item(db, item, next_status, reviewer_id, now,
                             answer=answer, answerability=answerability, evidence_ids=ev_json, notes=notes,
                             review_data=review_data)
        if status == "submitted":
            peers = db.all(
                "SELECT * FROM expert_reviews WHERE scope = ? AND question_id = ? AND role = ? ORDER BY reviewer_id",
                (scope, qid, batch),
            )
            if len(peers) == 3 and all(p["status"] == "submitted" for p in peers):
                if _consensus_check(peers, json.loads(item["original"])):
                    _advance_item(
                        db, item, "verified", "consensus", now,
                        answer=peers[0]["answer"], answerability=peers[0]["answerability"],
                        evidence_ids=peers[0]["evidence_ids"], notes=_combined_notes(peers), verified_at=now,
                        review_data=json.loads(peers[0].get("review_data") or "{}"),
                    )
                else:
                    _advance_item(
                        db, item, "flagged", "disagreement", now,
                        notes="Kết quả 3 chuyên gia có khác biệt, cần đối chiếu thống nhất.", verified_at=None,
                    )
    elif not allocation["requires_audit"]:
        final_status = "verified" if status == "submitted" else status
        _advance_item(
            db, item, final_status, reviewer_id, now,
            answer=answer, answerability=answerability, evidence_ids=ev_json, notes=notes,
            verified_at=now if status == "submitted" else None, review_data=review_data,
        )
    else:
        item_status = "draft" if status == "submitted" else status
        _advance_item(
            db, item, item_status, reviewer_id, now,
            answer=answer, answerability=answerability, evidence_ids=ev_json, notes=notes, verified_at=None,
            review_data=review_data,
        )
        if status == "submitted":
            db.execute(
                "UPDATE expert_reviews SET status = 'pending', revision = revision + 1, updated_at = ? "
                "WHERE scope = ? AND question_id = ? AND role = 'auditor'",
                (now, scope, qid),
            )


def submit_audit_decision(db, item, erow, auditor_id, revision, decision, audit_note):
    """Approve or reject a primary expert's audited question."""
    now = timestamp()
    if erow["role"] != "auditor":
        raise ReviewError("Câu hỏi này không thuộc hàng đợi kiểm định chéo.")
    if erow["revision"] != revision or erow["status"] in ("approved", "rejected"):
        raise ReviewConflict("Trạng thái kiểm định đã thay đổi. Hãy tải lại trang.")
    primary = db.locked(
        "SELECT * FROM expert_reviews WHERE scope = ? AND question_id = ? AND role = 'primary'",
        (item["scope"], item["question_id"]),
    )
    if not primary or primary["status"] != "submitted":
        raise ReviewConflict("Người chấm chính chưa nộp bản hoàn chỉnh để kiểm định.")

    note = audit_note.strip()
    if decision == "reject" and not note:
        raise ReviewError("Hãy ghi rõ lý do lỗi để trả lại người chấm chính.")
    if decision not in ("approve", "reject"):
        raise ReviewError("Thao tác kiểm định không hợp lệ.")

    auditor_status = "approved" if decision == "approve" else "rejected"
    changed = db.execute(
        "UPDATE expert_reviews SET status = ?, revision = ?, notes = ?, review_note = ?, updated_at = ? "
        "WHERE scope = ? AND question_id = ? AND role = 'auditor' AND reviewer_id = ? AND revision = ?",
        (auditor_status, revision + 1, note, "" if decision == "approve" else note, now,
         erow["scope"], erow["question_id"], auditor_id, revision),
    ).rowcount
    if changed != 1:
        raise ReviewConflict("Trạng thái kiểm định đã thay đổi. Hãy tải lại trang.")

    if decision == "approve":
        notes_parts = [p for p in (primary["notes"].strip(), f"[Audit]: {note}" if note else "") if p]
        db.execute(
            "UPDATE expert_reviews SET review_note = '' WHERE scope = ? AND question_id = ? AND role = 'primary'",
            (item["scope"], item["question_id"]),
        )
        _advance_item(
            db, item, "verified", auditor_id, now,
            answer=primary["answer"], answerability=primary["answerability"],
            evidence_ids=primary["evidence_ids"], notes="\n\n".join(notes_parts), verified_at=now,
            review_data=json.loads(primary.get("review_data") or "{}"),
        )
    else:
        db.execute(
            "UPDATE expert_reviews SET status = 'needs_revision', revision = revision + 1, "
            "review_note = ?, updated_at = ? WHERE scope = ? AND question_id = ? AND role = 'primary'",
            (note, now, item["scope"], item["question_id"]),
        )
        _advance_item(
            db, item, "flagged", auditor_id, now,
            answer=primary["answer"], answerability=primary["answerability"],
            evidence_ids=primary["evidence_ids"], notes=note, verified_at=None,
        )


def adjudicate_item(db, scope, question_id, revision, answer, answerability, evidence_ids, notes, *, review_data=None, actor_id="admin"):
    """Finalize a calibration or overlap item after 3-expert disagreement while preserving expert_reviews."""
    now = timestamp()
    item = db.locked("SELECT * FROM review_items WHERE scope = ? AND question_id = ?", (scope, question_id))
    if not item or item["revision"] != revision:
        raise ReviewConflict("Trạng thái câu hỏi đã thay đổi. Hãy tải lại trang.")
    _advance_item(
        db, item, "verified", actor_id, now,
        answer=answer, answerability=answerability, evidence_ids=evidence_ids, notes=notes, verified_at=now,
        review_data=review_data,
    )


def scientific_dashboard_data(db, scope):
    """Gather progress across batches, per-expert queues, and Fleiss' Kappa metrics for admin dashboard."""
    slots = db.all(
        "SELECT es.slot, u.id, u.full_name, u.profession FROM expert_slots es "
        "JOIN reviewers u ON u.id = es.reviewer_id WHERE es.scope = ? ORDER BY es.slot",
        (scope,),
    )
    if not slots:
        questions = db.all("SELECT id, original FROM questions ORDER BY position")
        preview = summarize_allocation(allocate_scientific_batches(questions)) if len(questions) >= 5 else None
        return {"active": False, "preview": preview, "slots": [], "batches": {}, "kappa": None}

    rows = db.all(
        "SELECT er.reviewer_id, er.role, er.status, er.question_id, er.answerability, er.evidence_ids, qa.requires_audit "
        "FROM expert_reviews er LEFT JOIN question_allocations qa ON qa.question_id = er.question_id "
        "WHERE er.scope = ?",
        (scope,),
    )
    items = db.all(
        "SELECT r.status, qa.batch_type, qa.requires_audit FROM review_items r "
        "JOIN question_allocations qa ON qa.question_id = r.question_id WHERE r.scope = ?",
        (scope,),
    )
    primary_submitted = {
        row["question_id"] for row in rows if row["role"] == "primary" and row["status"] == "submitted"
    }

    for slot_info in slots:
        rid = slot_info["id"]
        mine = [r for r in rows if r["reviewer_id"] == rid]
        slot_info["counts"] = {
            role: {
                "total": sum(1 for r in mine if r["role"] == role),
                "done": sum(1 for r in mine if r["role"] == role and r["status"] in ("submitted", "approved")),
                "needs_revision": sum(1 for r in mine if r["role"] == role and r["status"] == "needs_revision"),
                "ready": sum(
                    1 for r in mine
                    if r["role"] == role and role == "auditor"
                    and r["question_id"] in primary_submitted and r["status"] in ("pending", "draft")
                ),
            }
            for role in ("calibration", "overlap", "primary", "auditor")
        }

    batches = {}
    for batch in ("calibration", "overlap", "independent"):
        subset = [row for row in items if row["batch_type"] == batch]
        batches[batch] = {
            "total": len(subset),
            "verified": sum(1 for r in subset if r["status"] == "verified"),
            "flagged": sum(1 for r in subset if r["status"] == "flagged"),
        }
    audit_subset = [row for row in items if row["batch_type"] == "independent" and row["requires_audit"]]
    batches["audit"] = {
        "total": len(audit_subset),
        "verified": sum(1 for r in audit_subset if r["status"] == "verified"),
        "flagged": sum(1 for r in audit_subset if r["status"] == "flagged"),
    }

    overlap_questions = db.all(
        "SELECT q.id, q.original FROM questions q JOIN question_allocations qa ON qa.question_id = q.id "
        "WHERE qa.batch_type = 'overlap'"
    )
    question_chunks = {}
    for q in overlap_questions:
        _, _, sources = review_content(json.loads(q["original"]))
        question_chunks[q["id"]] = [src["CHUNK_ID"] for src in sources]

    overlap_reviews = [r for r in rows if r["role"] == "overlap"]
    kappa = compute_fleiss_kappa(len(overlap_questions), overlap_reviews, question_chunks)
    return {"active": True, "preview": None, "slots": slots, "batches": batches, "kappa": kappa}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import the VacxiS review dataset into the configured database.")
    parser.add_argument("--upgrade-eval", action="store_true", help="Upgrade a matching existing eval import in place.")
    parser.add_argument("--migrate", action="store_true", help="Apply compatible schema additions to the current database.")
    parser.add_argument("--documents", type=Path, help="Import packaged document chunks.")
    parser.add_argument("dataset", type=Path, nargs="?")
    args = parser.parse_args()
    try:
        if args.migrate:
            with connect(database_location(), write=True) as db:
                ensure_schema(db)
            print("Đã cập nhật cấu trúc cơ sở dữ liệu.")
        elif args.documents:
            count = import_documents(database_location(), args.documents)
            print(f"Đã nạp {count} đoạn tài liệu.")
        elif args.dataset:
            count = import_dataset(database_location(), args.dataset, upgrade_eval=args.upgrade_eval)
            print(f"Đã chuẩn bị {count} câu cho hai lượt duyệt.")
        else:
            parser.error("Chọn dataset, --migrate hoặc --documents.")
    except (ReviewError, OSError) as error:
        parser.exit(1, f"{error}\n")
