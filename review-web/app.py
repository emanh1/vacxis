"""VacxiS reviewer and administrator web routes."""

from base64 import urlsafe_b64encode
import hmac
import json
import os
import re
import secrets
import time
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import BadData, URLSafeSerializer
from starlette.datastructures import FormData
from starlette.middleware.sessions import SessionMiddleware

from content import (
    ADMIN_STATUSES, ANSWERABILITY, BATCH_LABELS, PROFESSIONS, STATUS,
    changed_content, export_record, remove_citations, render_chunk, review_content, review_data, reviewer_label,
)
from dataset_format import as_golden, is_golden
from documents import packaged_documents
from calibration import calibration_state, comparison, consensus_fields, resolve_consensus
from store import (
    BASE_DIR, CONSENT_VERSION, EXPORT_CONSENT_VERSION, SCOPES, ReviewConflict, ReviewError, accept_invite, adjudicate_item,
    connect, database_location, digest, expert_slot_for, invite, invited_reviewer, issue_link,
    release_unstarted, reopen, reserve_operation, reset_scientific_review, save_expert_review, save_review,
    scientific_dashboard_data, setup_scientific_review, submit_audit_decision, timestamp, update_profile,
)


INVITE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
INVITE_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}")
INVITE_CODE = re.compile(rf"[{INVITE_ALPHABET}]{{12}}")
INVITE_ALIASES = str.maketrans({"O": "0", "I": "1", "L": "1"})
REVIEW_START = "/review/onboarding/start"


def token_for_code(code, key):
    signed = hmac.digest(key, f"vacxis-invite-code-v1:{code}".encode(), "sha256")
    return urlsafe_b64encode(signed).decode().rstrip("=")


def new_invitation(key):
    code = "".join(secrets.choice(INVITE_ALPHABET) for _ in range(12))
    return "-".join((code[:4], code[4:8], code[8:])), token_for_code(code, key)


def invitation_token(credential, key):
    if not isinstance(credential, str) or len(credential) > 2048:
        return None
    value = credential.strip()
    token = value.rsplit("#", 1)[-1]
    if INVITE_TOKEN.fullmatch(token):
        return token
    code = re.sub(r"[-\s]", "", value).upper().translate(INVITE_ALIASES)
    if INVITE_CODE.fullmatch(code):
        return token_for_code(code, key)
    return None


def redirect(path):
    return RedirectResponse(path, status_code=303)


def equal(left, right):
    return isinstance(left, str) and isinstance(right, str) and secrets.compare_digest(left.encode(), right.encode())


async def checked_form(request: Request):
    form = await request.form(max_fields=30, max_files=0)
    if not equal(form.get("csrf"), request.session.get("csrf")):
        raise HTTPException(403, "Phiên làm việc hết hạn. Hãy tải lại trang rồi thử lại.")
    return form


PostForm = Annotated[FormData, Depends(checked_form)]


def integer(value, message):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ReviewError(message) from None


def profile_from(form):
    profile = {field: str(form.get(field, "")).strip() or None for field in ("full_name", "profession", "workplace")}
    if profile["profession"] and profile["profession"] not in PROFESSIONS:
        raise ReviewError("Hãy chọn một vai trò chuyên môn trong danh sách.")
    if len(profile["full_name"] or "") > 120 or len(profile["workplace"] or "") > 200:
        raise ReviewError("Họ tên hoặc nơi công tác dài hơn giới hạn của biểu mẫu.")
    return profile


def needs_profile(person):
    return person["scope"] == "live" and person["consent"] is None


def reviewer(db, request, *, lock=False):
    identifier = request.session.get("reviewer_id")
    if not identifier:
        return None
    select = db.locked if lock else db.one
    person = select("SELECT * FROM reviewers WHERE id = ?", (identifier,))
    if person and person["access_version"] == request.session.get("access_version"):
        return person
    request.session.clear()
    return None


def require_reviewer(db, request, *, lock=False):
    person = reviewer(db, request, lock=lock)
    if not person:
        raise HTTPException(401, "Hãy mở lời mời hoặc liên hệ nhóm đồ án để đăng nhập lại.")
    return person


def is_admin(request):
    return request.session.get("admin_until", 0) > time.time()


def require_admin(request):
    if not is_admin(request):
        raise HTTPException(403, "Trang này dành cho nhóm đồ án.")


def get_item(db, person, identifier, *, lock=False):
    slot = expert_slot_for(db, person["id"])
    select = db.locked if lock else db.one
    if not slot:
        item = select(
            "SELECT r.*, q.query, q.original FROM review_items r JOIN questions q ON r.question_id = q.id "
            "WHERE r.scope = ? AND r.reviewer_id = ? AND r.question_id = ?",
            (person["scope"], person["id"], identifier),
        )
        if not item:
            raise HTTPException(404, "Câu hỏi này chưa được giao cho bạn.")
        return item, None

    erow = select(
        "SELECT * FROM expert_reviews WHERE scope = ? AND reviewer_id = ? AND question_id = ?",
        (person["scope"], person["id"], identifier),
    )
    if not erow:
        raise HTTPException(404, "Câu hỏi này chưa được giao cho bạn.")
    item = select(
        "SELECT r.*, q.query, q.original FROM review_items r JOIN questions q ON r.question_id = q.id "
        "WHERE r.scope = ? AND r.question_id = ?",
        (person["scope"], identifier),
    )
    alloc = db.one("SELECT * FROM question_allocations WHERE question_id = ?", (identifier,))
    primary_row = None
    if erow["role"] == "auditor":
        primary_row = db.one(
            "SELECT er.*, u.full_name, u.profession, u.consent, u.consent_version FROM expert_reviews er "
            "JOIN reviewers u ON u.id = er.reviewer_id "
            "WHERE er.scope = ? AND er.question_id = ? AND er.role = 'primary'",
            (person["scope"], identifier),
        )
    return item, {"slot": slot, "erow": erow, "alloc": alloc, "primary": primary_row}


def next_item(db, person, excluded=""):
    if expert_slot_for(db, person["id"]):
        available = db.one(
            "SELECT er.question_id FROM expert_reviews er JOIN questions q ON er.question_id = q.id "
            "LEFT JOIN question_allocations qa ON qa.question_id = er.question_id "
            "LEFT JOIN expert_reviews pr ON pr.scope = er.scope AND pr.question_id = er.question_id AND pr.role = 'primary' "
            "WHERE er.scope = ? AND er.reviewer_id = ? AND er.question_id <> ? AND ("
            "  (er.role = 'primary' AND er.status = 'needs_revision')"
            "  OR (er.role IN ('calibration', 'overlap', 'primary') AND er.status IN ('draft', 'pending'))"
            "  OR (er.role = 'auditor' AND er.status IN ('draft', 'pending') AND pr.status = 'submitted')"
            ") ORDER BY "
            "  CASE WHEN er.role = 'calibration' THEN 0 WHEN er.role = 'overlap' THEN 1 "
            "       WHEN er.role = 'primary' AND er.status IN ('pending', 'draft') AND qa.requires_audit = 1 THEN 2 "
            "       WHEN er.role = 'primary' AND er.status = 'needs_revision' THEN 3 "
            "       WHEN er.role = 'auditor' THEN 4 ELSE 5 END, "
            "  CASE WHEN er.status = 'draft' THEN 0 ELSE 1 END, q.position LIMIT 1",
            (person["scope"], person["id"], excluded),
        )
        if available:
            return available
        # Waiting audits are a fallback, including the audit just returned for revision.
        return db.one(
            "SELECT er.question_id FROM expert_reviews er JOIN questions q ON q.id = er.question_id "
            "JOIN expert_reviews pr ON pr.scope = er.scope AND pr.question_id = er.question_id AND pr.role = 'primary' "
            "WHERE er.scope = ? AND er.reviewer_id = ? AND er.role = 'auditor' "
            "AND er.status IN ('pending', 'draft', 'rejected') AND pr.status IN ('pending', 'draft', 'needs_revision') "
            "ORDER BY q.position LIMIT 1", (person["scope"], person["id"]),
        )
    return db.one(
        "SELECT r.question_id FROM review_items r JOIN questions q ON r.question_id = q.id "
        "WHERE r.scope = ? AND r.reviewer_id = ? AND r.question_id <> ? AND r.status IN ('draft', 'pending') "
        "ORDER BY CASE WHEN r.status = 'draft' THEN 0 ELSE 1 END, q.position LIMIT 1",
        (person["scope"], person["id"], excluded),
    )


def create_app(location=None):
    deployed = bool(os.environ.get("VERCEL"))
    location = location or database_location()
    secret = os.environ.get("REVIEW_SECRET_KEY", "")
    password = os.environ.get("REVIEW_ADMIN_PASSWORD", "")
    vercel_host = os.environ.get("VERCEL_BRANCH_URL") if os.environ.get("VERCEL_ENV") == "preview" else os.environ.get("VERCEL_PROJECT_PRODUCTION_URL")
    public_url = (os.environ.get("REVIEW_PUBLIC_URL") or (f"https://{vercel_host}" if vercel_host else "http://127.0.0.1:8000")).rstrip("/")
    origin = urlsplit(public_url)
    if not origin.hostname or origin.path or origin.query or origin.fragment or origin.username or origin.password:
        raise RuntimeError("REVIEW_PUBLIC_URL must be an HTTP(S) origin without a path.")
    if origin.scheme != "https" and (deployed or origin.scheme != "http" or origin.hostname not in {"127.0.0.1", "localhost"}):
        raise RuntimeError("Use HTTPS outside localhost.")
    if (secret and len(secret) < 32) or (password and len(password) < 24):
        raise RuntimeError("Use a REVIEW_SECRET_KEY of at least 32 characters and a REVIEW_ADMIN_PASSWORD of at least 24 characters.")
    if origin.scheme == "https" and (not secret or not password or secret == password):
        raise RuntimeError("Set distinct REVIEW_SECRET_KEY and REVIEW_ADMIN_PASSWORD values.")

    secret = secret or secrets.token_urlsafe(48)
    invitation_key = secret.encode()
    consent_signer = URLSafeSerializer(secret, salt="vacxis-consent-v1")
    assignment_signer = URLSafeSerializer(secret, salt="vacxis-scientific-reset-v1")
    calibration_signer = URLSafeSerializer(secret, salt="vacxis-calibration-v1")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(SessionMiddleware, secret_key=secret,
                       https_only=origin.scheme == "https", max_age=14 * 24 * 60 * 60, session_cookie="vacxis_review")
    if not deployed:
        app.mount("/assets", StaticFiles(directory=BASE_DIR / "public" / "assets"), name="assets")
    templates = Jinja2Templates(directory=BASE_DIR / "templates")

    def page(request, name, *, code=200, **context):
        return templates.TemplateResponse(request=request, name=f"{name}.html", status_code=code, context={
            "csrf": request.session.setdefault("csrf", secrets.token_urlsafe(32)),
            "admin": is_admin(request), "statuses": STATUS, "admin_statuses": ADMIN_STATUSES,
            "batch_labels": BATCH_LABELS, "scopes": SCOPES, "professions": PROFESSIONS,
            "labels": ANSWERABILITY, "reviewer_label": reviewer_label,
            "signed_in": bool(request.session.get("reviewer_id")) or is_admin(request),
            "consent_version": CONSENT_VERSION, **context,
        })

    @app.middleware("http")
    async def protect(request, call_next):
        if request.method == "POST" and (
            request.headers.get("sec-fetch-site") == "cross-site"
            or not request.headers.get("content-length", "").isdigit()
            or int(request.headers["content-length"]) > 131072
        ):
            response = HTMLResponse("Yêu cầu không hợp lệ hoặc quá dài.", status_code=400)
        else:
            response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; form-action 'self'; "
            "base-uri 'none'; frame-ancestors 'none'"
        )
        if not request.url.path.startswith("/assets/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        return page(request, "error", code=error.status_code, message=error.detail)

    @app.get("/")
    def home(request: Request):
        if is_admin(request):
            return redirect("/admin")
        if request.session.get("reviewer_id"):
            return redirect("/review")
        return page(request, "welcome")

    @app.get("/join")
    def join_page(request: Request):
        return page(request, "welcome", joining=True)

    def permission_intent(person, token):
        context = digest(token) if token else person["access_version"]
        return consent_signer.dumps([person["id"], person["profile_revision"], context, CONSENT_VERSION])

    def invitation_page(request, person, *, token=None, form=None, error=None, code=200):
        ask_profile = needs_profile(person)
        values = {field: form.get(field, "") for field in ("full_name", "profession", "workplace")} if form else {}
        return page(request, "invitation", joining=True, person=person, token=token,
                    ask_profile=ask_profile, profile_values=values,
                    intent=permission_intent(person, token) if ask_profile else None, error=error, code=code)

    def finish_profile(db, person, form, token=None):
        action = form.get("action")
        if (action not in {"share", "skip"}
                or not equal(form.get("permission_intent"), permission_intent(person, token))):
            raise ReviewConflict("Lựa chọn chia sẻ thông tin đã thay đổi. Hãy kiểm tra và thử lại.")
        sharing = action == "share"
        update_profile(db, person, "accepted" if sharing else "declined", profile_from(form) if sharing else {})

    def failed_invitation(request):
        return page(request, "welcome", joining=True, code=400,
                    error="Hãy kiểm tra mã hoặc liên kết. Nếu lời mời đã dùng hoặc hết hạn, hãy xin nhóm đồ án cấp lại.")

    @app.post("/join")
    def join(request: Request, form: PostForm):
        token = invitation_token(form.get("token"), invitation_key)
        action = form.get("action", "enter")
        if not token or action not in {"open", "enter", "begin", "share", "skip"}:
            return failed_invitation(request)
        with connect(location) as db:
            invited = invited_reviewer(db, token)
        if not invited:
            return failed_invitation(request)
        if action == "open" or (action == "enter" and needs_profile(invited)):
            return invitation_page(request, invited, token=token)
        following = None
        try:
            with connect(location, write=True) as db:
                person = accept_invite(db, token)
                if not person:
                    return failed_invitation(request)
                if needs_profile(person):
                    finish_profile(db, person, form, token)
                    following = next_item(db, person)
                elif action in {"share", "skip"}:
                    raise ReviewConflict("Lựa chọn chia sẻ thông tin đã thay đổi. Hãy kiểm tra và thử lại.")
        except ReviewError as error:
            with connect(location) as db:
                invited = invited_reviewer(db, token)
            if not invited:
                return failed_invitation(request)
            return invitation_page(request, invited, token=token, form=form, error=str(error),
                                   code=409 if isinstance(error, ReviewConflict) else 422)
        request.session.clear()
        request.session.update(reviewer_id=person["id"], access_version=person["access_version"])
        return redirect(f"/review/{following['question_id']}" if following else "/review")

    @app.get("/admin/login")
    def admin_login_page(request: Request):
        if is_admin(request):
            return redirect("/admin")
        return page(request, "welcome", login=True)

    @app.post("/admin/login")
    def admin_login(request: Request, form: PostForm):
        if not password or not equal(form.get("password"), password):
            return page(request, "welcome", login=True, code=403, error="Mật khẩu quản trị không đúng.")
        request.session.clear()
        request.session["admin_until"] = int(time.time()) + 8 * 60 * 60
        return redirect("/admin")

    @app.post("/logout")
    def logout(request: Request, form: PostForm):
        if request.session.get("reviewer_id"):
            with connect(location, write=True) as db:
                db.execute("UPDATE reviewers SET access_version = access_version + 1 WHERE id = ? AND access_version = ?",
                           (request.session["reviewer_id"], request.session.get("access_version")))
        request.session.clear()
        return redirect("/")

    def _parse_submission(form, original_json, *, require_content=True):
        answer = remove_citations(str(form.get("answer", "")))
        notes = str(form.get("notes", "")).strip()
        label = form.get("answerability")
        if len(answer) > 20000 or len(notes) > 3000 or label not in ANSWERABILITY:
            raise ReviewError("Kiểm tra độ dài câu trả lời, ghi chú và mức độ tài liệu trả lời được.")
        choices = set(form.getlist("evidence_id"))
        _, _, sources = review_content(json.loads(original_json))
        selected = [chunk["CHUNK_ID"] for chunk in sources if chunk["CHUNK_ID"] in choices]
        if label == "UNANSWERABLE":
            selected = []
        if require_content and (not answer or (label != "UNANSWERABLE" and not selected)):
            raise ReviewError("Hãy điền câu trả lời và chọn đoạn làm căn cứ nếu tài liệu trả lời được.")
        return answer, label, selected, notes

    def parse_review_data(form, original_json, answer, label, evidence, action):
        original = json.loads(original_json)
        original_query = original.get("input", original.get("QUERY", ""))
        query = str(form.get("query", original_query)).strip()
        question_reason = str(form.get("question_reason", "")).strip()
        answer_reason = str(form.get("answer_reason", "")).strip()
        if not query or len(query) > 10000 or max(len(question_reason), len(answer_reason)) > 3000:
            raise ReviewError("Kiểm tra độ dài câu hỏi và lý do chỉnh sửa.")
        reference, _, _ = review_content(original)
        query_changed = query != original_query.strip()
        answer_changed = answer != remove_citations(reference)
        if action == "confirm" and ((query_changed and not question_reason) or (answer_changed and not answer_reason)):
            raise ReviewError("Vui lòng ghi lý do cho từng phần câu hỏi hoặc câu trả lời đã chỉnh sửa.")
        result = None
        if action == "reject":
            result = "REJECTED"
        elif action == "confirm":
            result = "EDITED" if changed_content(original, query, answer, label, evidence) else "APPROVED"
        return {
            "query": query,
            "question_reason": question_reason if query_changed else "",
            "answer_reason": answer_reason if answer_changed else "",
            "status": result,
        }

    @app.get("/review")
    def review_list(request: Request, status: str = "", q: str = "", role: str = ""):
        with connect(location) as db:
            person = require_reviewer(db, request)
            if needs_profile(person):
                return redirect(REVIEW_START)
            slot = expert_slot_for(db, person["id"])
            if not slot:
                rows = db.all(
                    "SELECT r.question_id, r.status, r.review_data, q.query FROM review_items r JOIN questions q ON q.id = r.question_id "
                    "WHERE r.scope = ? AND r.reviewer_id = ? ORDER BY q.position",
                    (person["scope"], person["id"]),
                )
                next_row = next_item(db, person)
                expert_summary = None
            else:
                rows = db.all(
                    "SELECT er.question_id, er.status, er.role, er.review_note, er.review_data, q.query, "
                    "pr.status AS primary_status, pr.review_data AS primary_data "
                    "FROM expert_reviews er JOIN questions q ON q.id = er.question_id "
                    "LEFT JOIN expert_reviews pr ON pr.scope = er.scope AND pr.question_id = er.question_id "
                    "AND pr.role = 'primary' "
                    "WHERE er.scope = ? AND er.reviewer_id = ? "
                    "ORDER BY CASE WHEN er.status = 'needs_revision' THEN 0 ELSE 1 END, "
                    "CASE WHEN er.role = 'calibration' THEN 1 WHEN er.role = 'overlap' THEN 2 "
                    "WHEN er.role = 'primary' THEN 3 ELSE 4 END, q.position",
                    (person["scope"], person["id"]),
                )
                for r in rows:
                    r["audit_ready"] = (r["role"] != "auditor") or (r["primary_status"] == "submitted")
                expert_summary = {
                    "slot": slot["slot"],
                    "needs_revision": sum(1 for r in rows if r["status"] == "needs_revision"),
                    "calibration": {
                        "total": sum(1 for r in rows if r["role"] == "calibration"),
                        "done": sum(1 for r in rows if r["role"] == "calibration" and r["status"] == "submitted"),
                    },
                    "overlap": {
                        "total": sum(1 for r in rows if r["role"] == "overlap"),
                        "done": sum(1 for r in rows if r["role"] == "overlap" and r["status"] == "submitted"),
                    },
                    "primary": {
                        "total": sum(1 for r in rows if r["role"] == "primary"),
                        "done": sum(1 for r in rows if r["role"] == "primary" and r["status"] == "submitted"),
                    },
                    "auditor": {
                        "total": sum(1 for r in rows if r["role"] == "auditor"),
                        "done": sum(1 for r in rows if r["role"] == "auditor" and r["status"] == "approved"),
                        "ready": sum(1 for r in rows if r["role"] == "auditor" and r["audit_ready"] and r["status"] in ("pending", "draft")),
                    },
                }
                next_row = next_item(db, person)
        for row in rows:
            detail = json.loads(row.get("primary_data") or "{}") if row.get("role") == "auditor" else review_data(row)
            row["query"] = detail.get("query") or row["query"]
        search = q.strip()[:120].casefold()
        filtered = [
            row for row in rows
            if (not status or row["status"] == status)
            and (not role or (role == "needs_revision" and row["status"] == "needs_revision") or row.get("role") == role)
            and (search in row["query"].casefold() or search in row["question_id"].casefold())
        ]
        done = (
            sum(row["status"] in ("submitted", "approved") for row in rows)
            if slot else sum(row["status"] == "verified" for row in rows)
        )
        return page(request, "queue", person=person, rows=filtered, total=len(rows),
                    done=done, next_row=next_row, expert_summary=expert_summary, role=role,
                    status=status, search=q[:120], message=request.session.pop("message", ""))

    @app.get(REVIEW_START)
    def start_review(request: Request):
        with connect(location) as db:
            person = require_reviewer(db, request)
        return invitation_page(request, person) if needs_profile(person) else redirect("/review")

    @app.post(REVIEW_START)
    def start_review_submit(request: Request, form: PostForm):
        try:
            with connect(location, write=True) as db:
                person = require_reviewer(db, request, lock=True)
                if not needs_profile(person):
                    return redirect("/review")
                finish_profile(db, person, form)
                following = next_item(db, person)
        except ReviewError as error:
            return invitation_page(request, person, form=form, error=str(error),
                                   code=409 if isinstance(error, ReviewConflict) else 422)
        return redirect(f"/review/{following['question_id']}" if following else "/review")

    def question_page(request, person, item, expert_ctx=None, *, form=None, error=None, code=200):
        original = json.loads(item["original"])
        reference, original_label, sources = review_content(original)
        reference = remove_citations(reference)
        if expert_ctx:
            erow = expert_ctx["erow"]
            is_auditor = erow["role"] == "auditor"
            source_row = expert_ctx["primary"] if is_auditor else erow
            data = review_data(source_row)
            answer = source_row["answer"] if source_row["answer"] is not None else reference
            label = source_row["answerability"] or original_label
            selected = json.loads(source_row["evidence_ids"]) if source_row["evidence_ids"] is not None else [c["CHUNK_ID"] for c in sources]
            notes = erow["notes"]
            locked = (
                (erow["status"] in ("approved", "rejected") or expert_ctx["primary"]["status"] != "submitted")
                if is_auditor else (erow["status"] in ("submitted", "approved"))
            )
            view_item = dict(item, status=erow["status"], revision=erow["revision"])
            audit_ctx = {
                "slot": expert_ctx["slot"]["slot"],
                "role": erow["role"],
                "batch_type": expert_ctx["alloc"]["batch_type"],
                "stratum": expert_ctx["alloc"]["stratum"],
                "requires_audit": expert_ctx["alloc"]["requires_audit"],
                "review_note": erow["review_note"],
                "locked": locked,
                "is_auditor": is_auditor,
                "primary_ready": expert_ctx["primary"]["status"] == "submitted" if is_auditor else True,
                "waiting": is_auditor and erow["status"] in ("pending", "draft", "rejected") and expert_ctx["primary"]["status"] != "submitted",
                "primary_notes": expert_ctx["primary"]["notes"] if is_auditor else "",
                "primary_reviewer": (
                    reviewer_label(expert_ctx["primary"])
                    if (is_auditor and expert_ctx["primary"].get("consent") == "accepted"
                        and (expert_ctx["primary"].get("consent_version") or 0) >= CONSENT_VERSION)
                    else (f"Người duyệt {expert_ctx['primary']['reviewer_id']}" if is_auditor else "")
                ),
            }
        else:
            data = review_data(item)
            answer = item["answer"] if item["answer"] is not None else reference
            label = item["answerability"] or original_label
            selected = json.loads(item["evidence_ids"]) if item["evidence_ids"] is not None else [c["CHUNK_ID"] for c in sources]
            notes = item["notes"]
            view_item = dict(item)
            audit_ctx = {"locked": item["status"] == "verified", "is_auditor": False}
        if selected and label == "UNANSWERABLE":
            selected = []
        values = {
            "answer": answer, "answerability": label, "evidence_ids": selected, "notes": notes, "audit_note": notes,
            "query": data.get("query") or item["query"],
            "question_reason": data.get("question_reason", ""), "answer_reason": data.get("answer_reason", ""),
        }
        if form is not None:
            values.update({field: str(form.get(field, values.get(field, ""))) for field in
                           ("query", "answer", "answerability", "notes", "audit_note", "question_reason", "answer_reason")})
            values["evidence_ids"] = form.getlist("evidence_id")
        result_status = data.get("status") if view_item["status"] in ("verified", "submitted", "approved") else None
        rendered_sources = [dict(render_chunk(chunk), doc_id=chunk["CHUNK_ID"].rsplit("_c_", 1)[0]) for chunk in sources]
        return page(request, "question", person=person, item=view_item, values=values,
                    audit_ctx=audit_ctx, original_answer=reference, original_query=item["query"], result_status=result_status,
                    sources=rendered_sources, error=error, code=code,
                    message=request.session.pop("message", ""))

    @app.get("/review/{identifier}")
    def question(request: Request, identifier: str):
        with connect(location) as db:
            person = require_reviewer(db, request)
            item, expert_ctx = get_item(db, person, identifier)
            if needs_profile(person):
                return redirect(REVIEW_START)
        return question_page(request, person, item, expert_ctx)

    def document_page(request, db, item):
        original = json.loads(item["original"])
        golden = as_golden(original)
        doc_ids = golden["additional_metadata"]["doc_ids"] if golden else original.get("RELEVANT_DOCS", [])
        documents = []
        for doc_id in dict.fromkeys(doc_ids):
            rows = db.all("SELECT * FROM document_chunks WHERE doc_id = ? ORDER BY position, chunk_id", (doc_id,))
            if not rows:
                rows = packaged_documents().get(doc_id, [])
            if not rows:
                documents.append({"id": doc_id, "title": doc_id, "categories": [], "missing": True})
                continue
            groups = {}
            previous_header = None
            for row in rows:
                category = row["category"] or "Thông tin khác"
                header = "" if (category, row["header"]) == previous_header else row["header"]
                groups.setdefault(category, []).append(dict(
                    render_chunk({"CHUNK_ID": row["chunk_id"], "TEXT": row["text"]}), header=header,
                ))
                previous_header = (category, row["header"])
            documents.append({
                "id": doc_id, "title": rows[0]["doc_title"],
                "categories": [{"name": name.capitalize(), "chunks": chunks} for name, chunks in groups.items()],
            })
        return templates.TemplateResponse(request=request, name="documents.html", context={"documents": documents})

    @app.get("/review/{identifier}/documents")
    def documents(request: Request, identifier: str):
        with connect(location) as db:
            person = require_reviewer(db, request)
            item, _ = get_item(db, person, identifier)
            if needs_profile(person):
                raise HTTPException(403, "Vui lòng hoàn tất bước bắt đầu duyệt.")
            return document_page(request, db, item)

    @app.get("/review/{identifier}/audit-next")
    def audit_next(request: Request, identifier: str):
        with connect(location) as db:
            person = require_reviewer(db, request)
            _, expert = get_item(db, person, identifier)
            if needs_profile(person) or not expert or expert["erow"]["role"] != "auditor":
                raise HTTPException(403, "Trang này dành cho người kiểm định câu hỏi.")
            following = next_item(db, person)
            return {"next": f"/review/{following['question_id']}" if following and (
                following["question_id"] != identifier or expert["primary"]["status"] == "submitted"
            ) else None}

    def shared_state(db, token):
        try:
            generation = calibration_signer.loads(token) if len(token) <= 2048 else None
        except BadData:
            generation = None
        if (not isinstance(generation, list) or len(generation) != 3
                or not isinstance(generation[0], str) or generation[0] not in SCOPES):
            raise HTTPException(404, "Liên kết đối chiếu không hợp lệ.")
        state = calibration_state(db, generation[0])
        if generation != state["generation"] or not state["ready"]:
            raise HTTPException(404, "Liên kết đối chiếu chưa sẵn sàng hoặc không còn hiệu lực.")
        return generation[0], state

    def shared_item(db, scope, identifier):
        item = db.one(
            "SELECT r.*, q.query, q.original FROM review_items r JOIN questions q ON q.id = r.question_id "
            "JOIN question_allocations qa ON qa.question_id = q.id "
            "WHERE r.scope = ? AND r.question_id = ? AND qa.batch_type = 'calibration' AND r.status = 'flagged'",
            (scope, identifier),
        )
        if not item:
            raise ReviewConflict("Câu này đã được chốt hoặc không thuộc danh sách cần đối chiếu. Hãy mở lại liên kết chung.")
        return item

    def calibration_page(request, db, token, scope, state, *, identifier=None, form=None, error=None, code=200):
        pending = state["pending"]
        identifier = identifier or (pending[0]["question_id"] if pending else None)
        item = shared_item(db, scope, identifier) if identifier else None
        context = {}
        if item:
            experts, groups, note, signature = comparison(db, scope, item)
            _, _, chunks = review_content(json.loads(item["original"]))
            context = {"item": item, "groups": groups, "comparison_note": note, "expert_signature": signature,
                       "fields": consensus_fields(experts, form), "notes": str(form.get("notes", "")) if form else "",
                       "sources": [dict(render_chunk(c), doc_id=c["CHUNK_ID"].rsplit("_c_", 1)[0]) for c in chunks]}
        return page(request, "calibration", token=token, scope=scope, collection=state, error=error, code=code,
                    joining=True, signed_in=False, **context)

    @app.get("/calibration/{token}")
    def calibration_collection(request: Request, token: str, question: str = ""):
        with connect(location) as db:
            scope, state = shared_state(db, token)
            if question and question not in {r["question_id"] for r in state["pending"]}:
                return redirect(f"/calibration/{token}")
            return calibration_page(request, db, token, scope, state, identifier=question or None)

    @app.get("/calibration/{token}/documents/{identifier}")
    def calibration_documents(request: Request, token: str, identifier: str):
        with connect(location) as db:
            scope, _ = shared_state(db, token)
            try:
                item = shared_item(db, scope, identifier)
            except ReviewConflict as exc:
                raise HTTPException(404, str(exc)) from None
            return document_page(request, db, item)

    @app.post("/calibration/{token}")
    def calibration_submit(request: Request, token: str, form: PostForm):
        identifier = str(form.get("question_id", ""))
        try:
            with connect(location, write=True) as db:
                # Use the generation lock shared with scientific reset before the item lock.
                db.locked("SELECT id FROM dataset WHERE id = 1")
                scope, state = shared_state(db, token)
                db.locked("SELECT question_id FROM review_items WHERE scope = ? AND question_id = ?", (scope, identifier))
                item = shared_item(db, scope, identifier)
                experts, _, _, signature = comparison(db, scope, item)
                revision = integer(form.get("revision"), "Phiên bản câu hỏi không hợp lệ.")
                if item["revision"] != revision or not equal(form.get("expert_signature"), signature):
                    raise ReviewConflict("Nội dung đã thay đổi ở phiên khác. Hãy tải lại và kiểm tra trước khi chốt.")
                resolved = resolve_consensus(consensus_fields(experts, form), form)
                action = resolved.get("result_action")
                if action not in ("confirm", "reject"):
                    raise ReviewError("Hãy chọn kết quả thống nhất.")
                answer, label, evidence, notes = _parse_submission(resolved, item["original"], require_content=action == "confirm")
                data = parse_review_data(resolved, item["original"], answer, label, evidence, action)
                data["consensus_choices"] = {f["key"]: f["choice"] if not f["agreed"] else "agreed" for f in consensus_fields(experts, form)}
                adjudicate_item(db, scope, identifier, revision, answer, label, json.dumps(evidence), notes,
                                review_data=data, actor_id="calibration_representative")
        except ReviewConflict as exc:
            with connect(location) as db:
                scope, state = shared_state(db, token)
                return calibration_page(request, db, token, scope, state, error=str(exc), code=409)
        except ReviewError as exc:
            with connect(location) as db:
                scope, state = shared_state(db, token)
                return calibration_page(request, db, token, scope, state, identifier=identifier, form=form, error=str(exc), code=422)
        return redirect(f"/calibration/{token}")

    @app.post("/review/{identifier}")
    def review_save(request: Request, identifier: str, form: PostForm):
        action = form.get("action")
        expert_ctx = None
        try:
            with connect(location, write=True) as db:
                person = require_reviewer(db, request, lock=True)
                item, expert_ctx = get_item(db, person, identifier, lock=True)
                if needs_profile(person):
                    return redirect(REVIEW_START)
                revision = integer(form.get("revision"), "Tải lại trang trước khi lưu.")
                if expert_ctx and expert_ctx["erow"]["role"] == "auditor":
                    if action not in ("audit_approve", "audit_reject"):
                        raise ReviewError("Hãy chọn thao tác kiểm định chéo.")
                    decision = "approve" if action == "audit_approve" else "reject"
                    submit_audit_decision(db, item, expert_ctx["erow"], person["id"], revision, decision,
                                          str(form.get("audit_note", "")))
                    status = "approved" if decision == "approve" else "rejected"
                    following = next_item(db, person, identifier)
                elif expert_ctx:
                    erow = expert_ctx["erow"]
                    actions = {"draft": "draft", "flag": "flagged", "confirm": "submitted", "reject": "submitted"}
                    if action not in actions:
                        raise ReviewError("Hãy chọn một thao tác duyệt.")
                    status = actions[action]
                    answer, label, selected, notes = _parse_submission(
                        form, item["original"], require_content=(action == "confirm"),
                    )
                    if status == "flagged" and not notes:
                        raise ReviewError("Hãy ghi lý do cần xem lại.")
                    data = parse_review_data(form, item["original"], answer, label, selected, action)
                    save_expert_review(db, item, erow, expert_ctx["alloc"], person["id"], revision,
                                       status, answer, label, selected, notes, review_data=data)
                    following = next_item(db, person, identifier) if status == "submitted" else None
                else:
                    if item["revision"] != revision or item["status"] == "verified":
                        raise ReviewConflict("Câu này vừa được cập nhật. Hãy mở lại để xem bản mới.")
                    actions = {"draft": "draft", "flag": "flagged", "confirm": "verified", "reject": "verified"}
                    if action not in actions:
                        raise ReviewError("Hãy chọn một thao tác duyệt.")
                    status = actions[action]
                    answer, label, selected, notes = _parse_submission(
                        form, item["original"], require_content=(action == "confirm"),
                    )
                    if status == "flagged" and not notes:
                        raise ReviewError("Hãy ghi lý do cần xem lại.")
                    data = parse_review_data(form, item["original"], answer, label, selected, action)
                    save_review(db, item, person["id"], revision, status, answer, label, selected, notes, review_data=data)
                    following = next_item(db, person, identifier) if status == "verified" else None
        except ReviewConflict as error:
            with connect(location) as db:
                item, expert_ctx = get_item(db, person, identifier)
            return question_page(request, person, item, expert_ctx, error=str(error), code=409)
        except ReviewError as error:
            return question_page(request, person, item, expert_ctx, form=form, error=str(error), code=422)
        request.session["message"] = f"{STATUS[status]} · {identifier}"
        if following:
            return redirect(f"/review/{following['question_id']}")
        return redirect("/review" if status in ("verified", "submitted", "approved", "rejected") else f"/review/{identifier}")

    @app.get("/quyen-rieng-tu")
    def privacy(request: Request):
        with connect(location) as db:
            person = reviewer(db, request) if request.session.get("reviewer_id") else None
        return page(request, "privacy", person=person, email=os.environ.get("REVIEW_PRIVACY_EMAIL", ""),
                    message=request.session.pop("message", ""))

    @app.post("/quyen-rieng-tu")
    def privacy_update(request: Request, form: PostForm):
        with connect(location, write=True) as db:
            person = require_reviewer(db, request, lock=True)
            if person["scope"] != "live":
                raise HTTPException(403, "Lượt thử nghiệm không có hồ sơ thông tin cá nhân.")
            if str(form.get("profile_revision", "")) != str(person["profile_revision"]):
                return page(request, "privacy", person=person, code=409, email=os.environ.get("REVIEW_PRIVACY_EMAIL", ""),
                            error="Hồ sơ vừa thay đổi. Hãy xem lại thông tin hiện tại trước khi lưu.")
            action = form.get("action")
            if action == "withdraw":
                update_profile(db, person, "withdrawn", {})
                request.session["message"] = "Đã xóa thông tin cá nhân khỏi hồ sơ. Bạn vẫn có thể duyệt bằng mã người duyệt."
            elif action == "share":
                if form.get("consent_version") != str(CONSENT_VERSION):
                    return page(request, "privacy", person=person, code=409, email=os.environ.get("REVIEW_PRIVACY_EMAIL", ""),
                                error="Nội dung xin phép vừa thay đổi. Hãy đọc lại trước khi lưu.")
                try:
                    update_profile(db, person, "accepted", profile_from(form))
                except ReviewError as error:
                    return page(request, "privacy", person=person, error=str(error), code=422,
                                email=os.environ.get("REVIEW_PRIVACY_EMAIL", ""))
                request.session["message"] = "Đã lưu thông tin theo lựa chọn của bạn."
            else:
                raise HTTPException(400, "Thao tác không hợp lệ.")
        return redirect("/quyen-rieng-tu")

    @app.get("/gioi-thieu")
    def refer_page(request: Request):
        with connect(location) as db:
            person = require_reviewer(db, request)
        if not person["can_refer"] or person["scope"] != "live":
            raise HTTPException(403, "Tài khoản này chưa được giao giới thiệu đồng nghiệp.")
        return page(request, "referral", person=person)

    @app.post("/gioi-thieu")
    def refer(request: Request, form: PostForm):
        label = str(form.get("label", "")).strip()
        with connect(location, write=True) as db:
            person = require_reviewer(db, request, lock=True)
            if not person["can_refer"] or person["scope"] != "live":
                raise HTTPException(403, "Tài khoản này chưa được giao giới thiệu đồng nghiệp.")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", label):
                return page(request, "referral", person=person, code=422,
                            error="Nhập mã tối đa 40 ký tự, gồm chữ không dấu, số hoặc dấu gạch.")
            db.execute("INSERT INTO referrals (id, referred_by, label, created_at) VALUES (?, ?, ?, ?)",
                       (secrets.token_hex(16), person["id"], label, timestamp()))
        request.session["message"] = "Đã gửi đề xuất cho nhóm đồ án."
        return redirect("/review")

    def admin_page(request, scope, *, status="", batch="", invitation=None, scientific_invites=None, error=None):
        with connect(location) as db:
            rows = db.all(
                "SELECT r.question_id, r.status, r.revision, r.reviewer_id, r.verified_at, q.query, "
                "u.id, u.full_name, u.profession, qa.batch_type, qa.primary_slot, qa.auditor_slot, qa.requires_audit "
                "FROM review_items r JOIN questions q ON q.id = r.question_id "
                "LEFT JOIN reviewers u ON u.id = r.reviewer_id "
                "LEFT JOIN question_allocations qa ON qa.question_id = r.question_id "
                "WHERE r.scope = ? ORDER BY q.position", (scope,),
            )
            people = db.all(
                "SELECT u.id, u.full_name, u.profession, u.workplace FROM reviewers u WHERE u.scope = ? "
                "AND (EXISTS (SELECT 1 FROM review_items r WHERE r.reviewer_id = u.id) "
                "     OR EXISTS (SELECT 1 FROM expert_reviews er WHERE er.reviewer_id = u.id)) ORDER BY u.id", (scope,),
            )
            referrals = db.all("SELECT * FROM referrals WHERE handled_at IS NULL ORDER BY created_at") if scope == "live" else []
            scientific = scientific_dashboard_data(db, scope)
            calibration = calibration_state(db, scope)
        code, token = invitation or (None, None)
        filtered = [
            row for row in rows
            if (not status or row["status"] == status)
            and (not batch or (batch == "audit" and row["requires_audit"]) or row["batch_type"] == batch)
        ]
        return page(request, "admin", scope=scope, total=len(rows), people=people, referrals=referrals,
                    scientific=scientific, scientific_invites=scientific_invites,
                    calibration=calibration,
                    calibration_url=f"{public_url}/calibration/{calibration_signer.dumps(calibration['generation'])}" if calibration["ready"] else None,
                    available=sum(not row["reviewer_id"] for row in rows),
                    done=sum(row["status"] == "verified" for row in rows), status=status, batch=batch,
                    rows=filtered,
                    invite_operation=secrets.token_hex(16), reviewer_operation=secrets.token_hex(16),
                    setup_operation=secrets.token_hex(16),
                    reset_operation=secrets.token_hex(16),
                    reset_assignment=assignment_signer.dumps([scope, sorted(slot["id"] for slot in scientific["slots"])]),
                    invite_url=f"{public_url}/join#{token}" if token else None, invite_code=code,
                    public_url=public_url, error=error, code=422 if error else 200,
                    message=request.session.pop("message", ""))

    @app.get("/admin")
    def admin_dashboard(request: Request, scope: str = "live", status: str = "", batch: str = ""):
        require_admin(request)
        if scope not in SCOPES:
            raise HTTPException(404, "Lượt duyệt không tồn tại.")
        return admin_page(request, scope, status=status, batch=batch)

    @app.post("/admin/scientific-setup")
    def admin_scientific_setup(request: Request, form: PostForm):
        require_admin(request)
        scope = form.get("scope", "live")
        if scope not in SCOPES:
            raise HTTPException(400, "Lượt duyệt không hợp lệ.")
        credentials = {}
        try:
            with connect(location, write=True) as db:
                reserve_operation(db, form.get("setup_operation"))
                tokens = {}
                for slot in (1, 2, 3):
                    code, token = new_invitation(invitation_key)
                    tokens[slot] = token
                    credentials[slot] = {"slot": slot, "code": code, "url": f"{public_url}/join#{token}"}
                created = setup_scientific_review(db, scope, tokens)
                for entry in created:
                    credentials[entry["slot"]]["reviewer_id"] = entry["reviewer_id"]
        except ReviewError as error:
            return admin_page(request, scope, error=str(error))
        return admin_page(request, scope, scientific_invites=[credentials[s] for s in (1, 2, 3)])

    @app.post("/admin/scientific-reset")
    def admin_scientific_reset(request: Request, form: PostForm):
        require_admin(request)
        scope = form.get("scope", "live")
        if scope not in SCOPES:
            raise HTTPException(400, "Lượt duyệt không hợp lệ.")
        try:
            try:
                assignment = assignment_signer.loads(form.get("reset_assignment", ""))
            except BadData as error:
                raise ReviewError("Phân công không hợp lệ. Hãy tải lại trang rồi thử lại.") from error
            if not isinstance(assignment, list) or len(assignment) != 2 or assignment[0] != scope:
                raise ReviewError("Phân công không hợp lệ. Hãy tải lại trang rồi thử lại.")
            with connect(location, write=True) as db:
                reserve_operation(db, form.get("reset_operation"))
                reset_scientific_review(db, scope, assignment[1])
        except ReviewError as error:
            return admin_page(request, scope, error=str(error))
        request.session["message"] = "Đã thu hồi phân công 3 chuyên gia. Các lời mời cũ đã hết hiệu lực."
        return redirect(f"/admin?scope={scope}")

    @app.post("/admin/invite")
    def admin_invite(request: Request, form: PostForm):
        require_admin(request)
        scope = form.get("scope", "live")
        if scope not in SCOPES:
            raise HTTPException(400, "Lượt duyệt không hợp lệ.")
        try:
            with connect(location, write=True) as db:
                reserve_operation(db, form.get("operation"))
                count = integer(form.get("count"), "Nhập số câu cần phân công.")
                code, token = new_invitation(invitation_key)
                invite(db, scope, count, form.get("can_refer") == "yes", form.get("referral_id") or None, token=token)
        except ReviewError as error:
            return admin_page(request, scope, error=str(error))
        return admin_page(request, scope, invitation=(code, token))

    @app.post("/admin/reissue")
    @app.post("/admin/release")
    def manage_assignments(request: Request, form: PostForm):
        require_admin(request)
        reviewer_id = form.get("reviewer_id", "")
        invitation = None
        try:
            with connect(location, write=True) as db:
                person = db.locked("SELECT scope FROM reviewers WHERE id = ?", (reviewer_id,))
                if not person:
                    raise HTTPException(404, "Không tìm thấy người duyệt.")
                reserve_operation(db, form.get("operation"))
                if request.url.path == "/admin/release":
                    count = release_unstarted(db, reviewer_id)
                    request.session["message"] = f"Đã thu hồi {count} câu chưa bắt đầu để phân công lại."
                else:
                    has_work = (
                        db.one("SELECT question_id FROM review_items WHERE reviewer_id = ? LIMIT 1", (reviewer_id,))
                        or db.one("SELECT question_id FROM expert_reviews WHERE reviewer_id = ? LIMIT 1", (reviewer_id,))
                    )
                    if not has_work:
                        raise ReviewError("Người duyệt này không còn câu được giao.")
                    code, token = new_invitation(invitation_key)
                    issue_link(db, reviewer_id, token=token)
                    invitation = (code, token)
        except ReviewError as error:
            return admin_page(request, person["scope"], error=str(error))
        return admin_page(request, person["scope"], invitation=invitation)

    def _history_context(db, scope, identifier):
        item = db.one("SELECT r.*, q.query, q.original FROM review_items r JOIN questions q ON q.id = r.question_id "
                      "WHERE r.scope = ? AND r.question_id = ?", (scope, identifier))
        if not item:
            raise HTTPException(404, "Không tìm thấy câu hỏi.")
        history_rows = db.all(
            "SELECT e.*, u.id, u.full_name, u.profession FROM review_events e LEFT JOIN reviewers u ON u.id = e.actor_id "
            "WHERE e.scope = ? AND e.question_id = ? ORDER BY e.revision DESC", (scope, identifier),
        )
        for entry in history_rows:
            entry["snapshot"] = json.loads(entry["snapshot"])
        alloc = db.one("SELECT * FROM question_allocations WHERE question_id = ?", (identifier,))
        expert_rows = db.all(
            "SELECT er.*, es.slot, u.id, u.full_name, u.profession FROM expert_reviews er "
            "LEFT JOIN expert_slots es ON es.reviewer_id = er.reviewer_id "
            "LEFT JOIN reviewers u ON u.id = er.reviewer_id "
            "WHERE er.scope = ? AND er.question_id = ? ORDER BY er.role, es.slot",
            (scope, identifier),
        )
        for er in expert_rows:
            er["evidence_list"] = json.loads(er["evidence_ids"] or "[]")
            er["review_detail"] = review_data(er)
        reference, original_label, sources = review_content(json.loads(item["original"]))
        reference = remove_citations(reference)
        adjudicate_seed = next(
            (er for er in expert_rows if er["status"] == "submitted"),
            None,
        )
        detail = review_data(item) or (review_data(adjudicate_seed) if adjudicate_seed else {})
        adj_values = {
            "query": detail.get("query") or item["query"],
            "question_reason": detail.get("question_reason", ""),
            "answer_reason": detail.get("answer_reason", ""),
            "result_action": "reject" if detail.get("status") == "REJECTED" else "confirm",
            "answer": item["answer"] if item["answer"] is not None else (
                adjudicate_seed["answer"] if adjudicate_seed and adjudicate_seed["answer"] is not None else reference
            ),
            "answerability": item["answerability"] or (adjudicate_seed["answerability"] if adjudicate_seed else original_label),
            "evidence_ids": (
                json.loads(item["evidence_ids"]) if item["evidence_ids"] is not None
                else (adjudicate_seed["evidence_list"] if adjudicate_seed else [c["CHUNK_ID"] for c in sources])
            ),
            "notes": item["notes"] or "",
        }
        return {
            "item": item, "events": history_rows, "alloc": alloc, "expert_rows": expert_rows,
            "sources": [render_chunk(chunk) for chunk in sources], "adj_values": adj_values,
        }

    @app.get("/admin/history/{scope}/{identifier}")
    def history(request: Request, scope: str, identifier: str):
        require_admin(request)
        with connect(location) as db:
            ctx = _history_context(db, scope, identifier)
        return page(request, "history", scope=scope, message=request.session.pop("message", ""), **ctx)

    @app.post("/admin/adjudicate/{scope}/{identifier}")
    def admin_adjudicate(request: Request, scope: str, identifier: str, form: PostForm):
        require_admin(request)
        if scope not in SCOPES:
            raise HTTPException(404, "Lượt duyệt không tồn tại.")
        try:
            with connect(location, write=True) as db:
                item = db.one(
                    "SELECT r.*, q.original FROM review_items r JOIN questions q ON q.id = r.question_id "
                    "WHERE r.scope = ? AND r.question_id = ?", (scope, identifier),
                )
                if not item:
                    raise HTTPException(404, "Không tìm thấy câu hỏi.")
                revision = integer(form.get("revision"), "Hãy tải lại trang.")
                result_action = form.get("result_action", "confirm")
                if result_action not in ("confirm", "reject"):
                    raise ReviewError("Kết quả thống nhất không hợp lệ.")
                answer, label, selected, notes = _parse_submission(
                    form, item["original"], require_content=(result_action == "confirm"),
                )
                data = parse_review_data(form, item["original"], answer, label, selected, result_action)
                adjudicate_item(db, scope, identifier, revision, answer, label, selected, notes, review_data=data)
        except ReviewConflict as error:
            with connect(location) as db:
                ctx = _history_context(db, scope, identifier)
            return page(request, "history", scope=scope, error=str(error), code=409, **ctx)
        except ReviewError as error:
            with connect(location) as db:
                ctx = _history_context(db, scope, identifier)
            ctx["adj_values"] = {
                "query": str(form.get("query", item.get("query", ""))),
                "question_reason": str(form.get("question_reason", "")),
                "answer_reason": str(form.get("answer_reason", "")),
                "result_action": str(form.get("result_action", "confirm")),
                "answer": str(form.get("answer", "")),
                "answerability": str(form.get("answerability", "")),
                "evidence_ids": form.getlist("evidence_id"),
                "notes": str(form.get("notes", "")),
            }
            return page(request, "history", scope=scope, error=str(error), code=422, **ctx)
        request.session["message"] = f"Đã chốt kết quả thống nhất · {identifier}"
        return redirect(f"/admin/history/{scope}/{identifier}")

    @app.post("/admin/reopen")
    def admin_reopen(request: Request, form: PostForm):
        require_admin(request)
        scope, identifier = form.get("scope", ""), form.get("identifier", "")
        try:
            with connect(location, write=True) as db:
                reopen(db, scope, identifier, integer(form.get("revision"), "Hãy tải lại trang."),
                       unassign=form.get("action") == "unassign")
        except ReviewError as error:
            raise HTTPException(409, str(error)) from error
        return redirect(f"/admin/history/{scope}/{identifier}")

    @app.get("/admin/export/{scope}")
    def export(request: Request, scope: str):
        require_admin(request)
        if scope not in SCOPES:
            raise HTTPException(404, "Lượt duyệt không tồn tại.")
        with connect(location) as db:
            first = db.one("SELECT original FROM questions ORDER BY position LIMIT 1")
            original = json.loads(first["original"]) if first else {}
            golden = as_golden(original)
            metadata = None
            if is_golden(original):
                entry = db.one("SELECT metadata FROM dataset_metadata WHERE id = 1")
                metadata = json.loads(entry["metadata"])
            elif golden is not None:
                metadata = {"total_generated": db.one("SELECT COUNT(*) AS total FROM questions")["total"]}
            rows = db.all(
                "SELECT r.*, q.original, u.full_name, u.profession "
                "FROM review_items r JOIN questions q ON q.id = r.question_id "
                "LEFT JOIN reviewers u ON u.id = r.reviewer_id AND u.scope = 'live' "
                "AND u.consent = 'accepted' AND u.consent_version >= ? "
                "WHERE r.scope = ? AND r.status = 'verified' ORDER BY q.position", (EXPORT_CONSENT_VERSION, scope),
            )
            scientific = bool(db.one("SELECT 1 FROM expert_slots WHERE scope = ? LIMIT 1", (scope,)))
            if scientific:
                allocs = {row["question_id"]: row for row in db.all("SELECT * FROM question_allocations")}
                erows = db.all(
                    "SELECT er.question_id, er.role, er.reviewer_id, u.scope, u.consent, u.consent_version, "
                    "u.full_name, u.profession "
                    "FROM expert_reviews er JOIN reviewers u ON u.id = er.reviewer_id "
                    "LEFT JOIN expert_slots es ON es.reviewer_id = er.reviewer_id "
                    "WHERE er.scope = ? AND er.status IN ('submitted', 'approved') "
                    "ORDER BY CASE WHEN er.role = 'primary' THEN 0 ELSE 1 END, es.slot",
                    (scope,),
                )
                by_qid = {}
                for er in erows:
                    label = (
                        reviewer_label(er)
                        if (er["scope"] == "live" and er["consent"] == "accepted"
                            and (er["consent_version"] or 0) >= EXPORT_CONSENT_VERSION and er["full_name"])
                        else er["reviewer_id"]
                    )
                    by_qid.setdefault(er["question_id"], []).append(label)
                for row in rows:
                    alloc = allocs.get(row["question_id"])
                    if alloc:
                        row["review_batch"] = alloc["batch_type"]
                        row["audited"] = bool(alloc["requires_audit"])
                    if row["question_id"] in by_qid:
                        row["annotator_labels"] = by_qid[row["question_id"]]
        filename = "test_dataset_verified.json" if scope == "live" else "internal_review_test_only.json"
        records = [export_record(row) for row in rows]
        payload = {"metadata": metadata, "goldens": records} if metadata is not None else records
        content = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent="\t")
        return Response(content + "\n", media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    return app


app = create_app()
