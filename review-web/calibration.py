"""Build the shared calibration collection without changing independent reviews."""

import json

from starlette.datastructures import FormData

from content import PROFESSIONS, changed_content, review_data
from store import CONSENT_VERSION, ReviewError, digest


def calibration_state(db, scope):
    slots = db.all("SELECT slot, reviewer_id FROM expert_slots WHERE scope = ? ORDER BY slot", (scope,))
    dataset = db.one("SELECT source_hash FROM dataset WHERE id = 1")
    rows = db.all(
        "SELECT er.question_id, er.reviewer_id, er.status FROM expert_reviews er "
        "JOIN question_allocations qa ON qa.question_id = er.question_id "
        "WHERE er.scope = ? AND er.role = 'calibration' AND qa.batch_type = 'calibration'", (scope,),
    )
    members = {s["reviewer_id"] for s in slots}
    questions = {r["question_id"] for r in db.all("SELECT question_id FROM question_allocations WHERE batch_type = 'calibration'")}
    ready = (len(slots) == 3 and bool(questions) and len(rows) == 3 * len(questions)
             and all({r["reviewer_id"] for r in rows if r["question_id"] == q} == members for q in questions)
             and all(r["status"] == "submitted" for r in rows))
    pending = db.all(
        "SELECT r.question_id, r.revision, q.query FROM review_items r "
        "JOIN questions q ON q.id = r.question_id JOIN question_allocations qa ON qa.question_id = q.id "
        "WHERE r.scope = ? AND r.status = 'flagged' AND qa.batch_type = 'calibration' ORDER BY q.position", (scope,),
    )
    return {"ready": ready, "total": len(questions), "submitted": sum(r["status"] == "submitted" for r in rows),
            "generation": [scope, dataset["source_hash"] if dataset else "", [s["reviewer_id"] for s in slots]],
            "pending": pending}


def grouped(values):
    """Group exact opinions in their original expert-slot order."""
    groups = []
    for expert, value in values:
        group = next((g for g in groups if g["value"] == value), None)
        if group is None:
            group = {"key": str(expert["slot"]), "value": value, "names": [], "slots": []}
            groups.append(group)
        group["names"].append(expert["name"])
        group["slots"].append(expert["slot"])
    for group in groups:
        group["name"] = " và ".join(group["names"])
    return groups


def comparison(db, scope, item):
    rows = db.all(
        "SELECT er.*, es.slot, u.full_name, u.profession, u.consent, u.consent_version "
        "FROM expert_reviews er JOIN expert_slots es ON es.scope = er.scope AND es.reviewer_id = er.reviewer_id "
        "JOIN reviewers u ON u.id = er.reviewer_id WHERE er.scope = ? AND er.question_id = ? "
        "AND er.role = 'calibration' ORDER BY es.slot", (scope, item["question_id"]),
    )
    experts = []
    original = json.loads(item["original"])
    for row in rows:
        data = review_data(row)
        name = f"Expert {row['slot']}"
        if row["consent"] == "accepted" and (row["consent_version"] or 0) >= CONSENT_VERSION and row["full_name"]:
            title = PROFESSIONS.get(row["profession"], "") if row["profession"] != "other" else ""
            name = f"{title} {row['full_name']}".strip()
        query = data.get("query") or item["query"]
        evidence = sorted(json.loads(row["evidence_ids"] or "[]"))
        status = data.get("status") or ("EDITED" if changed_content(original, query, row["answer"], row["answerability"], evidence) else "APPROVED")
        experts.append({"slot": row["slot"], "name": name, "revision": row["revision"], "values": {
            "query": data.get("query") or item["query"], "question_reason": data.get("question_reason", ""),
            "answer": row["answer"] or "", "answer_reason": data.get("answer_reason", ""),
            "answerability": row["answerability"], "evidence_ids": evidence,
            "notes": row["notes"] or "", "status": status,
        }})
    groups = grouped([(e, e["values"]) for e in experts])
    note = ""
    if len(groups) == 2:
        outlier = next(g for g in groups if len(g["slots"]) == 1)
        note = f"Ý kiến của {outlier['name']} khác với các ý kiến còn lại"
    elif len(groups) == 3:
        note = "3 người có ý kiến khác nhau"
    signature = digest(json.dumps(experts, ensure_ascii=False, sort_keys=True))
    return experts, groups, note, signature


def consensus_fields(experts, form=None):
    fields = []
    for key, title, reason in (("query", "Câu hỏi thống nhất", "question_reason"),
                               ("answer", "Câu trả lời thống nhất", "answer_reason"),
                               ("answerability", "Đánh giá tài liệu", None),
                               ("evidence_ids", "Đoạn trích căn cứ được chọn", None),
                               ("result_action", "Kết quả thống nhất", None)):
        values = []
        for expert in experts:
            source = expert["values"]
            value = ("reject" if source["status"] == "REJECTED" else "confirm") if key == "result_action" else source[key]
            values.append((expert, {"content": value, "reason": source[reason] if reason else ""}))
        groups = grouped(values)
        choice = str(form.get(f"choice_{key}", "")) if form is not None else ""
        selected = next((g for g in groups if g["key"] == choice), None)
        agreed = len(groups) == 1
        empty = {"content": [] if key == "evidence_ids" else "", "reason": ""}
        value = groups[0]["value"] if agreed else selected["value"] if selected else empty
        if choice == "custom" and not agreed and form is not None:
            value = {"content": form.getlist("evidence_id") if key == "evidence_ids" else str(form.get(key, "")),
                     "reason": str(form.get(reason, "")) if reason else ""}
        fields.append({"key": key, "title": title, "reason_key": reason, "groups": groups,
                       "agreed": agreed, "choice": choice, "content": value["content"], "reason": value["reason"]})
    return fields


def resolve_consensus(fields, form):
    """Resolve expert choices on the server; never trust posted readonly fields."""
    pairs = [("notes", str(form.get("notes", "")))]
    for field in fields:
        if not field["agreed"] and field["choice"] not in {g["key"] for g in field["groups"]} | {"custom"}:
            raise ReviewError(f"Hãy chọn ý kiến cho mục {field['title']}.")
        value = field["content"]
        if field["key"] == "evidence_ids":
            pairs.extend(("evidence_id", identifier) for identifier in value)
        else:
            pairs.append((field["key"], value))
        if field["reason_key"]:
            pairs.append((field["reason_key"], field["reason"]))
    return FormData(pairs)
