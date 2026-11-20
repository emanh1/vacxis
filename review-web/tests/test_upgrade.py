"""Protect shared calibration consensus, audit scheduling, and redeploy compatibility."""

import json
import os
import re
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import test_review
from app import create_app, next_item
from calibration import calibration_state, comparison, consensus_fields, grouped
from store import connect, ensure_schema


SECRET = "upgrade-tests-stable-secret-key-for-redeploy"


class UpgradeWorkflow(unittest.TestCase):
    _build_deepeval_dataset = test_review.ScientificReviewWorkflow._build_deepeval_dataset
    _setup_app = test_review.ScientificReviewWorkflow._setup_app
    _join_experts = test_review.ScientificReviewWorkflow._join_experts

    def start(self, count=60):
        with patch.dict(os.environ, {"REVIEW_SECRET_KEY": SECRET}):
            self._setup_app(count)
        self.experts = self._join_experts()
        self.guest = TestClient(self.app, follow_redirects=False)

    def submit_calibration(self, *, leave_last=False, reject=False):
        with connect(self.location) as db:
            ids = [r["question_id"] for r in db.all(
                "SELECT qa.question_id FROM question_allocations qa JOIN questions q ON q.id = qa.question_id "
                "WHERE qa.batch_type = 'calibration' ORDER BY q.position")]
        for index, qid in enumerate(ids):
            with connect(self.location) as db:
                original = json.loads(db.one("SELECT original FROM questions WHERE id = ?", (qid,))["original"])
            for slot, client in self.experts.items():
                if leave_last and index == len(ids) - 1 and slot == 3:
                    continue
                opened = client.get(f"/review/{qid}")
                changed = slot == 3
                result = client.post(f"/review/{qid}", data={
                    "csrf": test_review.csrf(opened), "revision": test_review.field(opened, "revision"),
                    "action": "reject" if reject and changed else "confirm",
                    "query": original["input"] + (" đã chỉnh" if changed else ""),
                    "question_reason": "Làm rõ câu hỏi" if changed else "",
                    "answer": original["expected_output"] + (" đã sửa" if changed else ""),
                    "answer_reason": "Làm rõ điều kiện" if changed else "",
                    "answerability": original["additional_metadata"]["answerability"],
                    "evidence_id": original["additional_metadata"]["source_chunk_ids"], "notes": "",
                })
                self.assertEqual(result.status_code, 303, result.text)
        return ids

    def link(self):
        dashboard = self.admin.get("/admin")
        match = re.search(r'id="calibration-url" readonly value="[^"]*(/calibration/[^"?]+)"', dashboard.text)
        self.assertIsNotNone(match, dashboard.text[:300])
        return match.group(1)

    def payload(self, page, *, slot=3):
        qid = test_review.field(page, "question_id")
        with connect(self.location) as db:
            item = db.one("SELECT r.*, q.query, q.original FROM review_items r JOIN questions q ON q.id = r.question_id WHERE r.scope = 'live' AND r.question_id = ?", (qid,))
            experts, _, _, _ = comparison(db, "live", item)
        fields = consensus_fields(experts)
        payload = {"csrf": test_review.csrf(page), "question_id": qid,
                   "revision": test_review.field(page, "revision"), "expert_signature": test_review.field(page, "expert_signature"), "notes": "Kết luận chung"}
        for field in fields:
            if not field["agreed"]:
                payload[f"choice_{field['key']}"] = next(g["key"] for g in field["groups"] if slot in g["slots"])
        return payload

    def test_gate_requires_every_submission_and_every_calibration_question(self):
        self.start()
        self.submit_calibration(leave_last=True)
        self.assertNotIn('id="calibration-url"', self.admin.get("/admin").text)
        with connect(self.location) as db:
            self.assertFalse(calibration_state(db, "live")["ready"])
        self.assertEqual(self.guest.get("/calibration/not-a-valid-token").status_code, 404)
        self.assertEqual(self.guest.get("/calibration/not-a-valid-token/documents/eval_0001").status_code, 404)

    def test_shared_collection_groups_opinions_preserves_reasons_and_resumes(self):
        self.start()
        ids = self.submit_calibration()
        with connect(self.location, write=True) as db:
            db.execute("UPDATE reviewers SET consent = 'declined' WHERE id = (SELECT reviewer_id FROM expert_slots WHERE scope = 'live' AND slot = 2)")
            originals = db.all("SELECT * FROM expert_reviews ORDER BY scope, question_id, reviewer_id")
        link = self.link()
        page = self.guest.get(link)
        self.assertEqual(page.status_code, 200, page.text)
        self.assertIn("Ý kiến của Bác sĩ Chuyên gia 3 khác với các ý kiến còn lại", page.text)
        self.assertIn("Bác sĩ Chuyên gia 1 và Expert 2", page.text)
        self.assertIn("Làm rõ câu hỏi", page.text)
        self.assertIn("Làm rõ điều kiện", page.text)
        self.assertNotIn("Người duyệt R-", page.text)
        self.assertNotIn("Kết quả 3 chuyên gia có khác biệt, cần đối chiếu thống nhất.", page.text)
        self.assertNotIn("Fleiss", page.text)
        self.assertNotIn("/admin/login", page.text)
        self.assertLess(page.text.index("Đối chiếu các bản đánh giá"), page.text.index("Đoạn trích căn cứ ("))
        self.assertLess(page.text.index("Đoạn trích căn cứ ("), page.text.index("<h2>Kết quả thống nhất"))
        self.assertRegex(page.text, r'<details class="disclosure calibration-sources">')
        first = test_review.field(page, "question_id")
        payload = self.payload(page)
        payload.update({"query": "Forged readonly value", "answer": "Forged readonly answer", "answer_reason": "Forged"})
        saved = self.guest.post(link, data=payload)
        self.assertEqual(saved.status_code, 303, saved.text)
        self.assertEqual(saved.headers["location"], link)
        self.assertNotEqual(test_review.field(self.guest.get(link), "question_id"), first)
        stale = self.guest.post(link, data=payload)
        self.assertEqual(stale.status_code, 409)
        with connect(self.location) as db:
            item = db.one("SELECT * FROM review_items WHERE scope = 'live' AND question_id = ?", (first,))
            self.assertTrue(item["answer"].endswith("đã sửa"))
            self.assertEqual(json.loads(item["review_data"])["answer_reason"], "Làm rõ điều kiện")
            self.assertEqual(db.all("SELECT * FROM expert_reviews ORDER BY scope, question_id, reviewer_id"), originals)
        for _ in range(len(ids) - 1):
            opened = self.guest.get(link)
            result = self.guest.post(link, data=self.payload(opened))
            self.assertEqual(result.status_code, 303, result.text)
        self.assertIn("Đã hoàn tất toàn bộ các câu cần đối chiếu", self.guest.get(link).text)
        exported = self.admin.get("/admin/export/live").json()["goldens"]
        self.assertEqual(len(exported), len(ids))
        self.assertEqual(exported[0]["additional_metadata"]["status"], "EDITED")
        self.assertEqual(exported[0]["additional_metadata"]["annotator_notes"]["lí do sửa câu hỏi"], "Làm rõ câu hỏi")

    def test_custom_consensus_validation_documents_scope_and_rejection(self):
        self.start()
        self.submit_calibration(reject=True)
        link = self.link()
        page = self.guest.get(link)
        qid = test_review.field(page, "question_id")
        with connect(self.location, write=True) as db:
            db.execute("INSERT INTO document_chunks (chunk_id, doc_id, doc_title, category, header, text, position) VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (f"{qid.replace('eval_', 'doc_')}_c_0000", qid.replace("eval_", "doc_"), "Tài liệu kiểm tra", "bảo quản", "Bảo quản", "Nhiệt độ bảo quản 2–8°C", 0))
            noncal = db.one("SELECT question_id FROM question_allocations WHERE batch_type <> 'calibration' LIMIT 1")["question_id"]
        docs = self.guest.get(f"{link}/documents/{qid}")
        self.assertIn("Nhiệt độ bảo quản", docs.text)
        self.assertEqual(self.guest.get(f"{link}/documents/{noncal}").status_code, 404)
        payload = self.payload(page)
        payload.update({"choice_query": "custom", "query": "Câu hỏi thống nhất khác", "question_reason": "",
                        "choice_answer": "custom", "answer": "Câu trả lời thống nhất khác", "answer_reason": "",
                        "choice_result_action": "custom", "result_action": "confirm"})
        invalid = self.guest.post(link, data=payload)
        self.assertEqual(invalid.status_code, 422)
        self.assertIn("Câu trả lời thống nhất khác", invalid.text)
        self.assertEqual(self.guest.post(link, data={**payload, "csrf": "wrong"}).status_code, 403)
        payload.update({"question_reason": "Làm rõ đối tượng", "answer_reason": "Thêm điều kiện"})
        self.assertEqual(self.guest.post(link, data=payload).status_code, 303)
        page = self.guest.get(link)
        rejected = self.guest.post(link, data=self.payload(page))
        self.assertEqual(rejected.status_code, 303, rejected.text)
        statuses = [g["additional_metadata"]["status"] for g in self.admin.get("/admin/export/live").json()["goldens"]]
        self.assertIn("REJECTED", statuses)

    def test_redeploy_and_migration_preserve_progress_accounts_and_shared_link(self):
        self.start()
        self.submit_calibration()
        link = self.link()
        with connect(self.location) as db:
            snapshot = {table: db.all(f"SELECT * FROM {table}") for table in (
                "reviewers", "invitations", "expert_slots", "expert_reviews", "review_items", "question_allocations")}
        with connect(self.location, write=True) as db:
            ensure_schema(db)
        with patch.dict(os.environ, {"REVIEW_SECRET_KEY": SECRET, "REVIEW_ADMIN_PASSWORD": test_review.PASSWORD}):
            updated = TestClient(create_app(self.location), follow_redirects=False)
        updated.cookies.update(self.experts[1].cookies)
        self.assertEqual(updated.get("/review").status_code, 200)
        self.assertEqual(updated.get(link).status_code, 200)
        self.assertEqual(updated.get("/review").status_code, 200)
        updated.cookies.update(self.admin.cookies)
        self.assertEqual(updated.get("/admin").status_code, 200)
        self.assertEqual(updated.get(link).status_code, 200)
        self.assertEqual(updated.get("/admin").status_code, 200)
        with connect(self.location) as db:
            for table, rows in snapshot.items():
                self.assertEqual(db.all(f"SELECT * FROM {table}"), rows)
        with connect(self.location, write=True) as db:
            db.execute("DELETE FROM expert_slots WHERE scope = 'live' AND slot = 3")
        self.assertEqual(self.guest.get(link).status_code, 404)

    def test_existing_name_consent_requires_voluntary_peer_sharing_and_withdraws(self):
        self.start()
        self.submit_calibration()
        with connect(self.location, write=True) as db:
            db.execute("UPDATE reviewers SET consent_version = 2 WHERE id = (SELECT reviewer_id FROM expert_slots WHERE scope = 'live' AND slot = 3)")
        link = self.link()
        opened = self.guest.get(link)
        self.assertIn("Ý kiến của Expert 3 khác", opened.text)
        self.assertNotIn("Bác sĩ Chuyên gia 3", opened.text)
        self.assertEqual(self.experts[3].get("/review").status_code, 200)
        self.assertEqual(self.guest.post(link, data=self.payload(opened)).status_code, 303)
        exported = self.admin.get("/admin/export/live").json()["goldens"]
        self.assertIn("Bác sĩ Chuyên gia 3", exported[0]["additional_metadata"]["ANNOTATOR_IDS"])
        profile = self.experts[3].get("/quyen-rieng-tu")
        self.assertIn("người có liên kết thảo luận Calibration", profile.text)
        payload = {"csrf": test_review.csrf(profile), "profile_revision": test_review.field(profile, "profile_revision"),
                   "action": "share", "consent_version": "2", "full_name": "Chuyên gia 3", "profession": "doctor", "workplace": "Riêng tư"}
        self.assertEqual(self.experts[3].post("/quyen-rieng-tu", data=payload).status_code, 409)
        payload["consent_version"] = test_review.field(profile, "consent_version")
        self.assertEqual(payload["consent_version"], "3")
        self.assertEqual(self.experts[3].post("/quyen-rieng-tu", data=payload).status_code, 303)
        opened = self.guest.get(link)
        self.assertIn("Bác sĩ Chuyên gia 3", opened.text)
        self.assertNotIn("Riêng tư", opened.text)
        stale = self.payload(opened)
        profile = self.experts[3].get("/quyen-rieng-tu")
        self.assertEqual(self.experts[3].post("/quyen-rieng-tu", data={
            "csrf": test_review.csrf(profile), "profile_revision": test_review.field(profile, "profile_revision"), "action": "withdraw",
        }).status_code, 303)
        self.assertEqual(self.guest.post(link, data=stale).status_code, 409)
        reopened = self.guest.get(link)
        self.assertNotIn("Bác sĩ Chuyên gia 3", reopened.text)
        self.assertIn("Ý kiến của Expert 3 khác", reopened.text)

    def test_audit_priority_skip_waiting_and_return_to_newly_ready_earlier_item(self):
        self.start()
        with connect(self.location, write=True) as db:
            person = db.one("SELECT u.* FROM reviewers u JOIN expert_slots es ON es.reviewer_id = u.id WHERE es.scope = 'live' AND es.slot = 2")
            rid = person["id"]
            db.execute("UPDATE expert_reviews SET status = CASE WHEN role = 'auditor' THEN 'approved' ELSE 'submitted' END WHERE scope = 'live'")
            rows = db.all("SELECT er.*, qa.requires_audit FROM expert_reviews er JOIN question_allocations qa ON qa.question_id = er.question_id JOIN questions q ON q.id = er.question_id WHERE er.scope = 'live' AND er.reviewer_id = ? ORDER BY q.position", (rid,))
            audited = next(r["question_id"] for r in rows if r["role"] == "primary" and r["requires_audit"])
            returned = next(r["question_id"] for r in rows if r["role"] == "primary" and r["question_id"] != audited and r["requires_audit"])
            normal = next(r["question_id"] for r in rows if r["role"] == "primary" and not r["requires_audit"])
            audits = [r["question_id"] for r in rows if r["role"] == "auditor"]
            self.assertGreaterEqual(len(audits), 2)
            db.execute("UPDATE expert_reviews SET status = 'pending' WHERE scope = 'live' AND reviewer_id = ? AND role = 'auditor'", (rid,))
            for qid in audits:
                db.execute("UPDATE expert_reviews SET status = 'pending' WHERE scope = 'live' AND question_id = ? AND role = 'primary'", (qid,))
            for qid, status in ((audited, "pending"), (returned, "needs_revision"), (normal, "draft")):
                db.execute("UPDATE expert_reviews SET status = ? WHERE scope = 'live' AND reviewer_id = ? AND question_id = ?", (status, rid, qid))
            db.execute("UPDATE expert_reviews SET status = 'submitted' WHERE scope = 'live' AND question_id = ? AND role = 'primary'", (audits[1],))
            self.assertEqual(next_item(db, person)["question_id"], audited)
            db.execute("UPDATE expert_reviews SET status = 'submitted' WHERE scope = 'live' AND reviewer_id = ? AND question_id = ?", (rid, audited))
            self.assertEqual(next_item(db, person)["question_id"], returned)
            db.execute("UPDATE expert_reviews SET status = 'submitted' WHERE scope = 'live' AND reviewer_id = ? AND question_id = ?", (rid, returned))
            self.assertEqual(next_item(db, person)["question_id"], audits[1])
            db.execute("UPDATE expert_reviews SET status = 'approved' WHERE scope = 'live' AND reviewer_id = ? AND question_id = ?", (rid, audits[1]))
            self.assertEqual(next_item(db, person)["question_id"], normal)
            db.execute("UPDATE expert_reviews SET status = 'submitted' WHERE scope = 'live' AND question_id = ? AND role = 'primary'", (audits[0],))
            self.assertEqual(next_item(db, person, normal)["question_id"], audits[0])
            db.execute("UPDATE expert_reviews SET status = 'approved' WHERE scope = 'live' AND reviewer_id = ? AND question_id = ?", (rid, audits[0]))
            db.execute("UPDATE expert_reviews SET status = 'submitted' WHERE scope = 'live' AND reviewer_id = ? AND question_id = ?", (rid, normal))
            waiting = next_item(db, person)["question_id"]
            self.assertIn(waiting, audits[2:])
            for qid in audits:
                if qid != waiting:
                    db.execute("UPDATE expert_reviews SET status = 'approved' WHERE scope = 'live' AND reviewer_id = ? AND question_id = ?", (rid, qid))
            db.execute("UPDATE expert_reviews SET status = 'rejected' WHERE scope = 'live' AND reviewer_id = ? AND question_id = ?", (rid, waiting))
            db.execute("UPDATE expert_reviews SET status = 'needs_revision' WHERE scope = 'live' AND role = 'primary' AND question_id = ?", (waiting,))
            self.assertEqual(next_item(db, person, waiting)["question_id"], waiting)
        opened = self.experts[2].get(f"/review/{waiting}")
        self.assertEqual(opened.status_code, 200, opened.text)
        self.assertIn("Chờ người chấm chính", opened.text)
        self.assertRegex(opened.text, r'value="audit_approve" disabled')
        self.assertIsNone(self.experts[2].get(f"/review/{waiting}/audit-next").json()["next"])
        with connect(self.location, write=True) as db:
            db.execute("UPDATE expert_reviews SET status = 'submitted' WHERE scope = 'live' AND role = 'primary' AND question_id = ?", (waiting,))
            db.execute("UPDATE expert_reviews SET status = 'pending' WHERE scope = 'live' AND reviewer_id = ? AND question_id = ?", (rid, waiting))
        self.assertEqual(self.experts[2].get(f"/review/{waiting}/audit-next").json()["next"], f"/review/{waiting}")


class OpinionGrouping(unittest.TestCase):
    def test_groups_preserve_slot_order_for_every_outlier_and_all_different(self):
        experts = [{"slot": slot, "name": f"Expert {slot}"} for slot in (1, 2, 3)]
        for outlier in (1, 2, 3):
            with self.subTest(outlier=outlier):
                groups = grouped([(expert, "other" if expert["slot"] == outlier else "same") for expert in experts])
                self.assertEqual(len(groups), 2)
                self.assertIn(1, groups[0]["slots"])
                self.assertEqual(next(g for g in groups if g["value"] == "other")["slots"], [outlier])
        groups = grouped([(expert, str(expert["slot"])) for expert in experts])
        self.assertEqual([g["name"] for g in groups], ["Expert 1", "Expert 2", "Expert 3"])
