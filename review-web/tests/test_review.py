"""Exercise the review workflow, access boundaries, and benchmark export."""

from collections import Counter
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import create_app
from content import remove_citations, render_chunk, review_content
from sampling import allocate_scientific_batches, compute_fleiss_kappa
from store import ReviewError, connect, ensure_schema, import_dataset, import_documents, invite


PASSWORD = "test-admin-password-for-review-web"
SOURCE_FILE = Path(__file__).resolve().parents[2] / "test_dataset.json"
FIRST = "/review/VAX_Q_0001"
WITH_TABLE = "/review/VAX_Q_0003"


def field(response, name):
    return re.search(rf'name="{name}" value="([A-Za-z0-9_.-]+)"', response.text).group(1)


def csrf(response):
    return field(response, "csrf")


def credentials(response):
    code = re.search(r'id="invite-code"[^>]* value="([A-Z0-9-]+)"', response.text).group(1)
    url = re.search(r'id="invite-url"[^>]* value="([^"]+)"', response.text).group(1)
    return code, url


class ReviewWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.location = str(Path(self.temp.name) / "review.sqlite3")
        dataset = json.loads(SOURCE_FILE.read_text(encoding="utf-8-sig"))[:3]
        fixture = Path(self.temp.name) / "fixture.json"
        fixture.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
        import_dataset(self.location, fixture)
        self.open_app()

    def open_app(self):
        with patch.dict(os.environ, {"REVIEW_ADMIN_PASSWORD": PASSWORD}, clear=False):
            self.app = create_app(self.location)
        self.admin = TestClient(self.app, follow_redirects=False)
        login = self.admin.get("/admin/login")
        self.admin_csrf = csrf(login)
        self.assertEqual(self.admin_post("/admin/login", {"password": PASSWORD}).status_code, 303)
        self.admin_csrf = csrf(self.admin.get("/admin"))

    def admin_post(self, path, data):
        return self.admin.post(path, data={"csrf": self.admin_csrf, **data})

    def create_invite(self, scope, count, **extra):
        operation = secrets.token_hex(16)
        response = self.admin_post("/admin/invite", {"scope": scope, "count": count, "operation": operation, **extra})
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def make_invite(self, scope="live", count=3, **extra):
        _, url = credentials(self.create_invite(scope, count, **extra))
        return url.rsplit("#", 1)[-1]

    def join(self, token, expected=303):
        client = TestClient(self.app, follow_redirects=False)
        response = self.open_invite(client, token, action="enter")
        if response.status_code == 200 and expected == 303:
            response = self.complete_invite(client, response, "skip")
        self.assertEqual(response.status_code, expected)
        return client

    def open_invite(self, client, token, action="open"):
        return client.post("/join", data={"token": token, "action": action, "csrf": csrf(client.get("/join"))})

    def complete_invite(self, client, page, action, **extra):
        data = {"csrf": csrf(page), "token": field(page, "token"), "action": action, **extra}
        if action in {"skip", "share"}:
            data["permission_intent"] = field(page, "permission_intent")
        return client.post("/join", data=data)

    def review(self, client, path, action, revision=0, **extra):
        data = {"answer": "Câu trả lời mẫu", "answerability": "UNANSWERABLE", "notes": "",
                "answer_reason": "Điều chỉnh câu trả lời mẫu.",
                "csrf": csrf(client.get(path)), "revision": revision, "action": action, **extra}
        return client.post(path, data=data)

    def test_deepeval_export_and_upgrade_preserve_review(self):
        temp = Path(self.temp.name)
        self.location = str(temp / "eval.sqlite3")
        contexts = [
            "[Tài liệu A | Chống chỉ định]\n[CHỦ ĐỀ: Vắc xin A] > [MỤC: CHỐNG CHỈ ĐỊNH]\nĐoạn đối chiếu A",
            "[Tài liệu B | Chống chỉ định]\n[CHỦ ĐỀ: Vắc xin B] > [MỤC: CHỐNG CHỈ ĐỊNH]\nĐoạn đối chiếu B",
        ]
        golden = {
            "input": "Câu hỏi DeepEval", "expected_output": "Đáp án gốc", "context": contexts,
            "additional_metadata": {
                "query_id": "eval_0001", "answerability": "ANSWERABLE", "complexity": "easy",
                "multi_hop_type": None, "query_style": "formal", "source_chunk_ids": ["a_c_0000", "b_c_0000"],
                "doc_ids": ["a", "b"], "doc_titles": ["Tài liệu A", "Tài liệu B"],
            },
        }
        converted = [{
            "QUERY_ID": "eval_0001", "QUERY": golden["input"], "REFERENCE_ANSWER": golden["expected_output"],
            "ANSWERABILITY": "ANSWERABLE", "QUERY_COMPLEXITY": "easy", "MULTI_HOP_TYPE": None,
            "QUERY_STYLE": "formal", "RELEVANT_DOCS": ["a", "b"], "ANNOTATOR_IDS": ["GEN"],
            "RELEVANT_CHUNKS": [{"CHUNK_ID": identifier, "TEXT": text, "RELEVANCE_SCORE": 3}
                                for identifier, text in zip(["a_c_0000", "b_c_0000"], contexts, strict=True)],
        }]
        old_path = temp / "converted.json"
        old_path.write_text(json.dumps(converted), encoding="utf-8")
        import_dataset(self.location, old_path)
        self.open_app()
        self.assertEqual(self.admin.get("/admin/export/live").json(), {"metadata": {"total_generated": 1}, "goldens": []})

        client = self.join(self.make_invite(count=1))
        path = "/review/eval_0001"
        page = client.get(path)
        self.assertIn("Đáp án gốc", page.text)
        self.assertIn("Đoạn đối chiếu B", page.text)
        self.assertIn('Đoạn trích trong tài liệu (2)', page.text)
        self.assertIn('<a href="#source-1"><strong>Vắc xin A</strong> · Chống chỉ định</a>', page.text)
        self.assertIn('<a href="#source-2"><strong>Vắc xin B</strong> · Chống chỉ định</a>', page.text)
        self.assertLess(page.text.index('href="#source-2"'), page.text.index('id="source-1"'))
        self.assertIn('<details class="disclosure source" open>', page.text)
        self.assertIn('<summary id="source-2"><strong>Vắc xin B</strong> · Chống chỉ định</summary>', page.text)
        self.assertIn('<div class="source-text"><p>Đoạn đối chiếu B</p>', page.text)
        self.assertIn('<pre>[Tài liệu B | Chống chỉ định]', page.text)
        self.assertEqual(page.text.count('href="#source-index"'), 2)
        self.assertEqual(self.review(client, path, "confirm", answerability="ANSWERABLE", evidence_id="b_c_0000",
                                     notes="Ghi chú cho nhóm").status_code, 303)
        response = self.admin.get("/admin/export/live")
        [exported] = response.json()["goldens"]
        self.assertEqual(set(exported), set(golden))
        self.assertNotIn("RELEVANCE_SCORE", response.text)
        self.assertEqual(exported["input"], golden["input"])
        self.assertEqual(exported["expected_output"], "Câu trả lời mẫu")
        self.assertEqual(exported["context"], contexts[1:])
        metadata = exported["additional_metadata"]
        self.assertEqual(metadata["source_chunk_ids"], ["b_c_0000"])
        self.assertEqual(metadata["doc_ids"], ["b"])
        self.assertEqual(metadata["doc_titles"], ["Tài liệu B"])
        self.assertTrue(metadata["VERIFIED"])
        self.assertTrue(metadata["ADJUDICATED"])
        self.assertEqual(metadata["VERIFICATION_NOTES"], "Ghi chú cho nhóm\n\nCâu trả lời sau chỉnh sửa:\nCâu trả lời mẫu")

        prepared_path = temp / "prepared.json"
        source_metadata = {"total_generated": 1, "model": "test-model"}
        unrelated = dict(golden, input="Câu hỏi khác")
        prepared_path.write_text(json.dumps({"metadata": source_metadata, "goldens": [unrelated]}), encoding="utf-8")
        with self.assertRaises(ReviewError):
            import_dataset(self.location, prepared_path, upgrade_eval=True)
        self.assertEqual(self.admin.get("/admin/export/live").json(), response.json())

        prepared_path.write_text(json.dumps({"metadata": source_metadata, "goldens": [golden]}), encoding="utf-8")
        with self.assertRaises(ReviewError):
            import_dataset(self.location, prepared_path)
        import_dataset(self.location, prepared_path, upgrade_eval=True)
        import_dataset(self.location, prepared_path)
        self.assertIn("Đoạn đối chiếu B", client.get(path).text)
        self.assertEqual(self.admin.get("/admin/export/live").json(), {"metadata": source_metadata, "goldens": [exported]})

    def test_invitation_can_be_entered_on_home_or_opened_as_link_once(self):
        home = TestClient(self.app, follow_redirects=False).get("/")
        self.assertIn('name="token"', home.text)
        self.assertIn('href="/admin/login"', home.text)
        self.assertEqual(self.admin.get("/admin/login").headers["location"], "/admin")

        issued = self.create_invite("live", 1)
        code, url = credentials(issued)
        self.assertRegex(code, r"^[0-9A-HJKMNP-TV-Z]{4}(?:-[0-9A-HJKMNP-TV-Z]{4}){2}$")
        self.assertIn("Sao chép lời mời", issued.text)
        entered = code.lower().replace("-", " ").translate(str.maketrans({"0": "o", "1": "l"}))
        client = self.join(f"  {entered}  ")
        self.assertEqual(client.get(FIRST).status_code, 200)
        self.join(url, expected=400)

        other_code, other_url = credentials(self.create_invite("live", 1))
        other = self.join(other_url)
        self.assertEqual(other.get(FIRST).status_code, 404)
        self.join(other_code, expected=400)

    def test_question_edits_reasons_drafts_and_rejection_export(self):
        client = self.join(self.make_invite(count=1))
        page = client.get(FIRST)
        self.assertIn("Đầu mục cần review", page.text)
        self.assertIn("Tổng kết đánh giá", page.text)
        self.assertIn("Tài liệu vô nghĩa", page.text)
        self.assertIn("Lưu chỉnh sửa", page.text)
        missing = self.review(client, FIRST, "confirm", query="Câu hỏi đã làm rõ", question_reason="", answer_reason="")
        self.assertEqual(missing.status_code, 422)
        self.assertIn("Câu hỏi đã làm rõ", missing.text)
        saved = self.review(client, FIRST, "draft", query="Câu hỏi đã làm rõ",
                            question_reason="", answer_reason="", notes="Ghi chú độc lập")
        self.assertEqual(saved.status_code, 303)
        with connect(self.location) as db:
            row = db.one("SELECT * FROM review_items WHERE scope = 'live' AND question_id = 'VAX_Q_0001'")
        self.assertIsNone(json.loads(row["review_data"])["status"])
        self.assertIn("Câu hỏi đã làm rõ", client.get(FIRST).text)
        result = self.review(client, FIRST, "confirm", revision=1, query="Câu hỏi đã làm rõ",
                             question_reason="Làm rõ đối tượng.", answer_reason="Bổ sung điều kiện.",
                             notes="Ghi chú độc lập")
        self.assertEqual(result.status_code, 303)
        [record] = self.admin.get("/admin/export/live").json()
        self.assertEqual(record["QUERY"], "Câu hỏi đã làm rõ")
        self.assertEqual(record["status"], "EDITED")
        self.assertEqual(record["annotator_notes"], {
            "lí do sửa câu hỏi": "Làm rõ đối tượng.", "lí do sửa câu trả lời": "Bổ sung điều kiện.",
            "ghi chú cho nhóm đồ án": "Ghi chú độc lập",
        })
        history = self.admin.get("/admin/history/live/VAX_Q_0001")
        self.assertIn("Làm rõ đối tượng.", history.text)
        self.assertIn("Bổ sung điều kiện.", history.text)
        self.assertEqual(self.review(client, FIRST, "confirm", revision=1).status_code, 409)

        self.assertEqual(self.admin_post("/admin/reopen", {
            "scope": "live", "identifier": "VAX_Q_0001", "revision": 2, "action": "reopen",
        }).status_code, 303)
        rejected = self.review(client, FIRST, "reject", revision=3, query="Câu hỏi đã làm rõ",
                               notes="Nguồn chưa phù hợp.", question_reason="Làm rõ đối tượng.")
        self.assertEqual(rejected.status_code, 303)
        [record] = self.admin.get("/admin/export/live").json()
        self.assertEqual(record["status"], "REJECTED")
        self.assertFalse(record["VERIFIED"])
        self.assertEqual(record["annotator_notes"]["ghi chú cho nhóm đồ án"], "Nguồn chưa phù hợp.")

    def test_notes_only_approval_and_source_document_access(self):
        client = self.join(self.make_invite(count=3))
        with connect(self.location) as db:
            original = json.loads(db.one("SELECT original FROM questions WHERE id = 'VAX_Q_0003'")["original"])
        answer, label, sources = review_content(original)
        doc_id = original["RELEVANT_DOCS"][0]
        fixture = Path(self.temp.name) / "documents.json"
        fixture.write_text(json.dumps([
            {"chunk_id": f"{doc_id}_c_900{i}", "doc_id": doc_id, "doc_title": "Tài liệu đầy đủ",
             "category": category, "header": f"Mục {i}", "text": text, "position": i}
            for i, (category, text) in enumerate([
                ("chống chỉ định", "Nội dung bổ sung A"), ("chống chỉ định", "Nội dung bổ sung B"),
                ("bảo quản", "<script>unsafe()</script><p>Bảo quản nguồn</p>"),
            ])
        ]), encoding="utf-8")
        import_documents(self.location, fixture)
        self.assertEqual(TestClient(self.app).get(WITH_TABLE + "/documents").status_code, 401)
        self.assertEqual(client.get("/review/not-assigned/documents").status_code, 404)
        full = client.get(WITH_TABLE + "/documents")
        self.assertEqual(full.status_code, 200)
        self.assertEqual(full.text.count('class="document-category"'), 2)
        self.assertIn("Nội dung bổ sung B", full.text)
        self.assertNotIn("<script>", full.text)
        values = {"answer": remove_citations(answer), "answerability": label, "notes": "Nguồn phù hợp."}
        if label != "UNANSWERABLE":
            values["evidence_id"] = [source["CHUNK_ID"] for source in sources]
        self.assertEqual(self.review(client, WITH_TABLE, "confirm", **values).status_code, 303)
        [record] = self.admin.get("/admin/export/live").json()
        self.assertEqual(record["status"], "APPROVED")
        self.assertEqual(record["annotator_notes"]["ghi chú cho nhóm đồ án"], "Nguồn phù hợp.")
        self.assertIsNone(record["annotator_notes"]["lí do sửa câu trả lời"])

    def test_schema_upgrade_preserves_existing_assignments(self):
        self.join(self.make_invite(count=1))
        with connect(self.location, write=True) as db:
            before = db.one("SELECT reviewer_id FROM review_items WHERE question_id = 'VAX_Q_0001' AND scope = 'live'")
            db.execute("ALTER TABLE review_items DROP COLUMN review_data")
            db.execute("ALTER TABLE expert_reviews DROP COLUMN review_data")
            ensure_schema(db)
            ensure_schema(db)
            after = db.one("SELECT reviewer_id, review_data FROM review_items WHERE question_id = 'VAX_Q_0001' AND scope = 'live'")
        self.assertEqual(before["reviewer_id"], after["reviewer_id"])
        self.assertEqual(json.loads(after["review_data"]), {})

    def test_reissued_invitation_replaces_previous_access_and_legacy_links_work(self):
        issued = self.create_invite("live", 1)
        previous_code, previous_url = credentials(issued)
        with connect(self.location) as db:
            reviewer = db.one("SELECT id FROM reviewers WHERE scope = 'live'")
        operation = re.findall(r'name="operation" value="([a-f0-9]{32})"', issued.text)[-1]
        replacement = self.admin_post("/admin/reissue", {"reviewer_id": reviewer["id"], "operation": operation})
        self.assertEqual(replacement.status_code, 200)
        code, _ = credentials(replacement)
        self.join(previous_code, expected=400)
        self.join(previous_url, expected=400)
        self.assertEqual(self.join(code).get(FIRST).status_code, 200)
        with connect(self.location, write=True) as db:
            legacy_token = invite(db, "test", 1, False, None)
        self.assertIn("Thử nghiệm nội bộ", self.join(legacy_token).get(FIRST).text)

    def test_live_permission_profile_and_withdrawal(self):
        token = self.make_invite()
        client = TestClient(self.app, follow_redirects=False)
        with patch("app.CONSENT_VERSION", 1):
            old_notice = self.open_invite(client, token)
        self.assertEqual(self.complete_invite(client, old_notice, "share", full_name="Tên nhập ở bản cũ").status_code, 409)
        notice = self.open_invite(client, token)
        self.assertEqual(notice.status_code, 200)
        self.assertIn("Nhóm đồ án xin phép", notice.text)
        self.assertIn("Bỏ qua và bắt đầu duyệt", notice.text)
        self.assertIn('name="full_name"', notice.text)
        self.assertNotIn('/admin/login', notice.text)
        self.assertEqual(client.get(FIRST).status_code, 401)
        self.assertEqual(self.admin.get("/admin/export/live").json(), [])

        result = self.complete_invite(client, notice, "share", full_name="Nguyễn Văn A", profession="doctor",
                                      workplace="Đơn vị kiểm thử")
        self.assertEqual(result.status_code, 303)
        self.assertEqual(result.headers["location"], FIRST)
        self.assertNotIn('name="full_name"', client.get(FIRST).text)
        self.assertEqual(self.review(client, FIRST, "confirm").status_code, 303)
        self.assertIn("Bác sĩ Nguyễn Văn A", client.get(FIRST).text)
        response = self.admin.get("/admin/export/live")
        self.assertNotIn("Đơn vị kiểm thử", response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")
        [record] = response.json()
        self.assertTrue(record["VERIFIED"])
        self.assertEqual(record["ANNOTATOR_IDS"][-1], "Bác sĩ Nguyễn Văn A")
        self.assertTrue(record["ADJUDICATED"])
        self.assertEqual(record["VERIFICATION_NOTES"], "Câu trả lời sau chỉnh sửa:\nCâu trả lời mẫu")
        self.assertEqual(record["RELEVANT_CHUNKS"], [])

        with connect(self.location, write=True) as db:
            db.execute("UPDATE reviewers SET consent_version = 1 WHERE scope = 'live'")
        [legacy] = self.admin.get("/admin/export/live").json()
        self.assertTrue(legacy["ANNOTATOR_IDS"][-1].startswith("R-"))
        privacy = client.get("/quyen-rieng-tu")
        self.assertIn("Đơn vị kiểm thử", privacy.text)
        renewal = {"csrf": csrf(privacy), "profile_revision": field(privacy, "profile_revision"), "action": "share",
                   "full_name": "Nguyễn Văn A", "profession": "doctor", "workplace": "Đơn vị kiểm thử"}
        self.assertEqual(client.post("/quyen-rieng-tu", data=renewal).status_code, 409)
        renewal["consent_version"] = field(privacy, "consent_version")
        self.assertEqual(client.post("/quyen-rieng-tu", data=renewal).status_code, 303)
        [renewed] = self.admin.get("/admin/export/live").json()
        self.assertEqual(renewed["ANNOTATOR_IDS"][-1], "Bác sĩ Nguyễn Văn A")

        privacy = client.get("/quyen-rieng-tu")
        profile_revision = field(privacy, "profile_revision")
        profile_action = {"csrf": csrf(privacy), "profile_revision": profile_revision}
        self.assertEqual(client.post("/quyen-rieng-tu", data={**profile_action, "action": "withdraw"}).status_code, 303)
        stale = client.post("/quyen-rieng-tu", data={**profile_action, "action": "share", "full_name": "Bản cũ"})
        self.assertEqual(stale.status_code, 409)
        self.assertNotIn("Nguyễn Văn A", self.admin.get("/admin/history/live/VAX_Q_0001").text)
        self.assertIn("Người duyệt R-", client.get(FIRST).text)
        [withdrawn] = self.admin.get("/admin/export/live").json()
        self.assertTrue(withdrawn["ANNOTATOR_IDS"][-1].startswith("R-"))

    def test_test_scope_verifies_without_profile_and_never_exports_live(self):
        token = self.make_invite("test")
        client = TestClient(self.app, follow_redirects=False)
        opened = self.open_invite(client, token)
        self.assertNotIn('name="full_name"', opened.text)
        self.assertNotIn('/admin/login', opened.text)
        self.assertEqual(self.complete_invite(client, opened, "begin").status_code, 303)
        self.assertIn("Thử nghiệm nội bộ", client.get(FIRST).text)
        result = self.review(client, FIRST, "confirm", full_name="Dữ liệu giả", profession="doctor")
        self.assertEqual(result.status_code, 303)
        self.assertNotIn('name="full_name"', client.get("/quyen-rieng-tu").text)
        self.assertEqual(self.admin.get("/admin/export/live").json(), [])
        self.assertEqual(len(self.admin.get("/admin/export/test").json()), 1)
        with connect(self.location) as db:
            self.assertIsNone(db.one("SELECT consent FROM reviewers WHERE scope = 'test'")["consent"])

    def test_assignment_link_reuse_revision_conflicts_and_reopen(self):
        token = self.make_invite(count=1)
        client = self.join(token)
        self.join(token, expected=400)
        self.assertEqual(client.get(WITH_TABLE).status_code, 404)
        self.assertEqual(client.get("/admin/export/live").status_code, 403)
        self.assertEqual(client.post(FIRST, data={"action": "confirm"}).status_code, 403)
        self.assertEqual(self.review(client, FIRST, "draft").status_code, 303)
        self.assertEqual(self.review(client, FIRST, "draft").status_code, 409)
        self.assertEqual(self.review(client, FIRST, "confirm", revision=1).status_code, 303)
        history = self.admin.get("/admin/history/live/VAX_Q_0001")
        self.assertIn("Lịch sử duyệt", history.text)
        response = self.admin_post("/admin/reopen", {"scope": "live", "identifier": "VAX_Q_0001", "revision": 2})
        self.assertEqual(response.status_code, 303)
        self.assertIn("Đang duyệt", client.get(FIRST).text)
        self.assertEqual(self.admin.get("/admin/export/live").json(), [])

    def test_return_confirmed_question_to_unassigned_pool(self):
        client = self.join(self.make_invite(count=1))
        self.assertEqual(self.review(client, FIRST, "confirm", answer="Bản duyệt nhầm", notes="Ghi chú thử").status_code, 303)
        self.assertEqual(len(self.admin.get("/admin/export/live").json()), 1)
        history_path = "/admin/history/live/VAX_Q_0001"
        self.assertIn("Trả về chưa phân công", self.admin.get(history_path).text)

        recovery = {"scope": "live", "identifier": "VAX_Q_0001", "revision": 1, "action": "unassign"}
        self.assertEqual(self.admin_post("/admin/reopen", recovery).status_code, 303)
        self.assertEqual(self.admin_post("/admin/reopen", recovery).status_code, 409)
        self.assertEqual(self.admin.get("/admin/export/live").json(), [])
        with connect(self.location) as db:
            available = db.one("SELECT COUNT(*) AS total FROM review_items WHERE scope = 'live' AND reviewer_id IS NULL")
        self.assertEqual(available["total"], 3)
        self.assertEqual(client.get(FIRST).status_code, 404)
        self.assertIn("Bản duyệt nhầm", self.admin.get(history_path).text)

        new_reviewer = self.join(self.make_invite(count=1))
        question = new_reviewer.get(FIRST)
        self.assertEqual(question.status_code, 200)
        self.assertNotIn("Bản duyệt nhầm", question.text)
        self.assertNotIn("Ghi chú thử", question.text)
        with connect(self.location) as db:
            assigned = db.one("SELECT reviewer_id FROM review_items WHERE scope = 'live' AND question_id = 'VAX_Q_0001'")
        dashboard = self.admin.get("/admin")
        operations = re.findall(r'name="operation" value="([a-f0-9]{32})"', dashboard.text)
        released = self.admin_post("/admin/release", {"reviewer_id": assigned["reviewer_id"], "operation": operations[-1]})
        self.assertEqual(released.status_code, 200)
        with connect(self.location) as db:
            available = db.one("SELECT COUNT(*) AS total FROM review_items WHERE scope = 'live' AND reviewer_id IS NULL")
        self.assertEqual(available["total"], 3)

    def test_table_preview_and_edited_export_keep_evidence_in_sync(self):
        client = self.join(self.make_invite())
        page = client.get(WITH_TABLE)
        self.assertIn('<div class="source-table" role="region" aria-label="Bảng trong tài liệu" tabindex="0"><table>',
                      page.text)
        self.assertIn('rowspan="2"', page.text)
        self.assertEqual(page.text.count('name="evidence_id"'), 1)
        chunk_id = "gardasil_9_001310178700_c_0009"
        review = {"answerability": "PARTIALLY_ANSWERABLE", "evidence_id": chunk_id}
        self.assertEqual(self.review(client, WITH_TABLE, "confirm", answerability="PARTIALLY_ANSWERABLE").status_code, 422)
        result = self.review(client, WITH_TABLE, "confirm", notes="Cần lưu ý cách diễn đạt.", **review)
        self.assertEqual(result.status_code, 303)
        exported = self.admin.get("/admin/export/live")
        self.assertIn('\n\t\t"QUERY_ID":', exported.text)
        self.assertIn('Câu trả lời mẫu', exported.text)
        [record] = exported.json()
        self.assertEqual(record["REFERENCE_ANSWER"], f"Câu trả lời mẫu [{chunk_id}]")
        self.assertEqual(record["CLAIM_EVIDENCE_MAP"], [])
        self.assertEqual(record["RELEVANT_DOCS"], ["gardasil_9_001310178700"])
        self.assertTrue(record["ADJUDICATED"])
        self.assertEqual(record["VERIFICATION_NOTES"],
                         "Cần lưu ý cách diễn đạt.\n\nCâu trả lời sau chỉnh sửa:\nCâu trả lời mẫu")

    def test_export_distinguishes_confirmation_from_notes_without_answer_edits(self):
        client = self.join(self.make_invite(count=1))
        with connect(self.location) as db:
            original = json.loads(db.one("SELECT original FROM questions WHERE id = 'VAX_Q_0001'")["original"])
        answer = original["REFERENCE_ANSWER"]
        self.assertEqual(self.review(client, FIRST, "confirm", answer=answer, notes="  ").status_code, 303)
        [confirmed] = self.admin.get("/admin/export/live").json()
        self.assertTrue(confirmed["VERIFIED"])
        self.assertFalse(confirmed["ADJUDICATED"])
        self.assertIsNone(confirmed["VERIFICATION_NOTES"])

        reopened = self.admin_post("/admin/reopen", {"scope": "live", "identifier": "VAX_Q_0001", "revision": 1})
        self.assertEqual(reopened.status_code, 303)
        result = self.review(client, FIRST, "confirm", revision=2, answer=answer, notes="Đối chiếu xong.")
        self.assertEqual(result.status_code, 303)
        [annotated] = self.admin.get("/admin/export/live").json()
        self.assertTrue(annotated["ADJUDICATED"])
        self.assertEqual(annotated["VERIFICATION_NOTES"], "Đối chiếu xong.")

    def test_referral_needs_admin_and_assigns_a_distinct_account(self):
        requester = self.join(self.make_invite(count=1, can_refer="yes"))
        consent = requester.get("/gioi-thieu")
        self.assertEqual(consent.status_code, 200)
        self.assertEqual(requester.post("/gioi-thieu", data={"csrf": csrf(consent), "label": "reviewer-2"}).status_code, 303)
        with connect(self.location) as db:
            referral = db.one("SELECT id FROM referrals")
        colleague = self.join(self.make_invite(count=1, referral_id=referral["id"]))
        self.assertEqual(colleague.get(FIRST).status_code, 404)
        self.assertEqual(colleague.get("/gioi-thieu").status_code, 403)

    def test_retaking_permission_in_another_tab_preserves_opt_out(self):
        token = self.make_invite()
        client = TestClient(self.app, follow_redirects=False)
        other = TestClient(self.app, follow_redirects=False)
        early = self.open_invite(client, token)
        late = self.open_invite(other, token)
        result = self.complete_invite(client, early, "skip")
        self.assertEqual(result.status_code, 303)
        self.assertEqual(result.headers["location"], FIRST)
        stale = self.complete_invite(other, late, "share", full_name="Bản cũ")
        self.assertEqual(stale.status_code, 400)
        self.assertEqual(self.admin.get("/admin/export/live").json(), [])
        with connect(self.location) as db:
            self.assertEqual(db.one("SELECT consent, full_name FROM reviewers"), {"consent": "declined", "full_name": None})

    def test_opening_live_invitation_keeps_admin_session_until_continue(self):
        token = self.make_invite()
        welcome = self.admin.get("/join")
        self.assertNotIn('/admin/login', welcome.text)
        notice = self.open_invite(self.admin, token)
        self.assertIn("chuyển phiên đăng nhập", notice.text)
        self.assertNotIn('/admin/login', notice.text)
        self.assertEqual(self.admin.get("/admin").status_code, 200)
        self.assertEqual(self.complete_invite(self.admin, notice, "skip").status_code, 303)
        self.assertEqual(self.admin.get("/admin").status_code, 403)

    def test_legacy_live_session_gets_permission_before_its_next_review(self):
        client = self.join(self.make_invite())
        with connect(self.location, write=True) as db:
            db.execute("UPDATE reviewers SET consent = NULL")
        path = "/review/onboarding/start"
        self.assertEqual(client.get(FIRST).headers["location"], path)
        start = client.get(path)
        self.assertIn('name="full_name"', start.text)
        old_form = client.post(FIRST, data={"csrf": csrf(start), "action": "confirm", "revision": 0,
                                            "answer": "Câu trả lời mẫu", "answerability": "UNANSWERABLE"})
        self.assertEqual(old_form.headers["location"], path)
        self.assertEqual(self.admin.get("/admin/export/live").json(), [])
        submitted = client.post(path, data={"csrf": csrf(start), "action": "skip",
                                           "permission_intent": field(start, "permission_intent")})
        self.assertEqual(submitted.headers["location"], FIRST)
        self.assertEqual(client.get(FIRST).status_code, 200)

    def test_invite_retry_does_not_reserve_again_and_release_revokes_link(self):
        operation = secrets.token_hex(16)
        payload = {"scope": "live", "count": 1, "operation": operation}
        invited = self.admin_post("/admin/invite", payload)
        token = re.search(r'/join#([A-Za-z0-9_-]{43})', invited.text).group(1)
        self.assertEqual(self.admin_post("/admin/invite", payload).status_code, 422)
        with connect(self.location) as db:
            [person] = db.all("SELECT id FROM reviewers")
        operations = re.findall(r'name="operation" value="([a-f0-9]{32})"', invited.text)
        self.assertEqual(self.admin_post("/admin/release", {"reviewer_id": person["id"], "operation": operations[-1]}).status_code, 200)
        self.join(token, expected=400)
        other = self.join(self.make_invite())
        self.assertEqual(other.get(FIRST).status_code, 200)

        profile = other.get("/quyen-rieng-tu")
        saved = other.post("/quyen-rieng-tu", data={"csrf": csrf(profile), "action": "share",
                           "profile_revision": field(profile, "profile_revision"), "full_name": "Tên sẽ được xóa",
                           "consent_version": field(profile, "consent_version")})
        self.assertEqual(saved.status_code, 303)
        with connect(self.location) as db:
            active = db.one("SELECT id FROM reviewers WHERE consent = 'accepted'")
        operations = re.findall(r'name="operation" value="([a-f0-9]{32})"', self.admin.get("/admin").text)
        self.assertEqual(self.admin_post("/admin/release", {"reviewer_id": active["id"], "operation": operations[-1]}).status_code, 200)
        self.assertEqual(other.get(FIRST).status_code, 401)
        with connect(self.location) as db:
            self.assertEqual(db.one("SELECT consent, full_name FROM reviewers WHERE id = ?", (active["id"],)),
                             {"consent": "withdrawn", "full_name": None})


class ScientificReviewWorkflow(unittest.TestCase):
    def _build_deepeval_dataset(self, count=12):
        strata = [
            ("ANSWERABLE", "SINGLE_HOP", "PATIENT_STYLE"),
            ("ANSWERABLE", "SINGLE_HOP", "CLINICAL_STYLE"),
            ("ANSWERABLE", "MULTI_HOP", "PATIENT_STYLE"),
            ("ANSWERABLE", "TABLE_REASONING", "CLINICAL_STYLE"),
            ("PARTIALLY_ANSWERABLE", "SINGLE_HOP", "PATIENT_STYLE"),
            ("PARTIALLY_ANSWERABLE", "TABLE_REASONING", "CLINICAL_STYLE"),
        ]
        goldens = []
        for idx in range(count):
            ans, comp, style = strata[idx % len(strata)]
            qid = f"eval_{idx + 1:04d}"
            doc_id = f"doc_{idx + 1:04d}"
            chunk_id = f"{doc_id}_c_0000"
            title = f"Tài liệu {idx + 1}"
            goldens.append({
                "input": f"Câu hỏi khoa học {idx + 1}",
                "expected_output": f"Đáp án chuẩn {idx + 1}",
                "context": [f"[{title} | Mục chính]\n[CHỦ ĐỀ: {title}] > [MỤC: CHỐNG CHỈ ĐỊNH]\nNội dung tham khảo {idx + 1}."],
                "additional_metadata": {
                    "query_id": qid,
                    "answerability": ans,
                    "complexity": comp,
                    "multi_hop_type": None,
                    "query_style": style,
                    "source_chunk_ids": [chunk_id],
                    "doc_ids": [doc_id],
                    "doc_titles": [title],
                },
            })
        return {"metadata": {"total_generated": count, "model": "test-model"}, "goldens": goldens}

    def _setup_app(self, count=12):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.location = str(Path(self.temp.name) / "scientific.sqlite3")
        dataset_path = Path(self.temp.name) / "eval.json"
        dataset_path.write_text(json.dumps(self._build_deepeval_dataset(count), ensure_ascii=False), encoding="utf-8")
        import_dataset(self.location, dataset_path)
        with patch.dict(os.environ, {"REVIEW_ADMIN_PASSWORD": PASSWORD}, clear=False):
            self.app = create_app(self.location)
        self.admin = TestClient(self.app, follow_redirects=False)
        self.admin.post("/admin/login", data={"csrf": csrf(self.admin.get("/admin/login")), "password": PASSWORD})
        self.admin_csrf = csrf(self.admin.get("/admin"))

    def _join_experts(self, scope="live"):
        dashboard = self.admin.get(f"/admin?scope={scope}")
        setup_op = field(dashboard, "setup_operation")
        res = self.admin.post("/admin/scientific-setup", data={
            "csrf": self.admin_csrf, "scope": scope, "setup_operation": setup_op,
        })
        self.assertEqual(res.status_code, 200, res.text)
        tokens = re.findall(r'id="invite-url-[123]" readonly value="[^#]+#([A-Za-z0-9_-]{43})"', res.text)
        self.assertEqual(len(tokens), 3)
        clients = {}
        for slot, token in enumerate(tokens, start=1):
            client = TestClient(self.app, follow_redirects=False)
            opened = client.post("/join", data={"csrf": csrf(client.get("/join")), "token": token, "action": "enter"})
            self.assertEqual(opened.status_code, 200)
            done = client.post("/join", data={
                "csrf": csrf(opened), "token": field(opened, "token"), "action": "share",
                "permission_intent": field(opened, "permission_intent"),
                "full_name": f"Chuyên gia {slot}", "profession": "doctor", "workplace": "Bệnh viện A",
            })
            self.assertEqual(done.status_code, 303)
            clients[slot] = client
        return clients

    def test_unused_scientific_assignment_can_be_withdrawn_and_recreated(self):
        self._setup_app(12)
        experts = self._join_experts("live")
        other_scope = self.admin.get("/admin?scope=test")
        self.assertEqual(self.admin.post("/admin/scientific-setup", data={
            "csrf": self.admin_csrf, "scope": "test", "setup_operation": field(other_scope, "setup_operation"),
        }).status_code, 200)
        with connect(self.location) as db:
            allocations = db.all("SELECT * FROM question_allocations ORDER BY question_id")
            old_ids = {row["reviewer_id"] for row in db.all("SELECT reviewer_id FROM expert_slots WHERE scope = 'live'")}
            old_question = db.one("SELECT question_id FROM expert_reviews WHERE scope = 'live' AND role = 'calibration' LIMIT 1")["question_id"]
            original = json.loads(db.one("SELECT original FROM questions WHERE id = ?", (old_question,))["original"])
        dashboard = self.admin.get("/admin?scope=live")
        payload = {"csrf": self.admin_csrf, "scope": "live", "reset_operation": field(dashboard, "reset_operation"),
                   "reset_assignment": field(dashboard, "reset_assignment")}
        stale_dashboard = self.admin.get("/admin?scope=live")
        stale_payload = {**payload, "reset_operation": field(stale_dashboard, "reset_operation")}
        history = self.admin.get(f"/admin/history/live/{old_question}")
        stale_adjudication = {"csrf": self.admin_csrf, "revision": field(history, "revision"),
                              "answer": original["expected_output"], "answerability": original["additional_metadata"]["answerability"],
                              "evidence_id": original["additional_metadata"]["source_chunk_ids"], "notes": ""}
        self.assertEqual(self.admin.post("/admin/scientific-reset", data={
            **payload, "reset_assignment": payload["reset_assignment"] + "x",
        }).status_code, 422)
        self.assertEqual(self.admin.post("/admin/scientific-reset", data={**payload, "scope": "test"}).status_code, 422)
        response = self.admin.post("/admin/scientific-reset", data=payload)
        self.assertEqual(response.status_code, 303)
        with connect(self.location) as db:
            self.assertEqual(db.one("SELECT COUNT(*) AS n FROM expert_slots WHERE scope = 'live'")["n"], 0)
            self.assertEqual(db.one("SELECT COUNT(*) AS n FROM expert_reviews WHERE scope = 'live'")["n"], 0)
            self.assertEqual(db.one("SELECT COUNT(*) AS n FROM review_items WHERE scope = 'live' AND reviewer_id IS NULL")["n"], 12)
            self.assertEqual(db.one("SELECT COUNT(*) AS n FROM expert_slots WHERE scope = 'test'")["n"], 3)
            self.assertEqual(db.all("SELECT * FROM question_allocations ORDER BY question_id"), allocations)
            for reviewer_id in old_ids:
                self.assertIsNone(db.one("SELECT id FROM reviewers WHERE id = ?", (reviewer_id,)))
                self.assertIsNone(db.one("SELECT token_hash FROM invitations WHERE reviewer_id = ?", (reviewer_id,)))
        self.assertEqual(experts[1].get(f"/review/{old_question}").status_code, 401)
        self.assertEqual(self.admin.post("/admin/scientific-reset", data=payload).status_code, 422)
        self.assertEqual(self.admin.post(f"/admin/adjudicate/live/{old_question}", data=stale_adjudication).status_code, 409)
        new_experts = self._join_experts("live")
        self.assertEqual(self.admin.post("/admin/scientific-reset", data=stale_payload).status_code, 422)
        with connect(self.location) as db:
            slots = db.all("SELECT reviewer_id FROM expert_slots WHERE scope = 'live'")
        self.assertEqual(len(slots), 3)
        self.assertTrue(old_ids.isdisjoint({row["reviewer_id"] for row in slots}))
        self.assertEqual(new_experts[1].get("/review").status_code, 200)
        self.assertEqual(self.admin.post(f"/admin/adjudicate/live/{old_question}", data=stale_adjudication).status_code, 409)

    def test_scientific_reset_preserves_saved_drafts(self):
        self._setup_app(12)
        experts = self._join_experts("live")
        with connect(self.location) as db:
            qid = db.one("SELECT question_id FROM expert_reviews WHERE scope = 'live' AND role = 'calibration' LIMIT 1")["question_id"]
        page = experts[1].get(f"/review/{qid}")
        saved = experts[1].post(f"/review/{qid}", data={
            "csrf": csrf(page), "revision": field(page, "revision"), "action": "draft",
            "answer": "Nội dung đang duyệt.", "answerability": "UNANSWERABLE", "notes": "Ghi chú đang làm.",
        })
        self.assertEqual(saved.status_code, 303)
        dashboard = self.admin.get("/admin?scope=live")
        result = self.admin.post("/admin/scientific-reset", data={
            "csrf": self.admin_csrf, "scope": "live", "reset_operation": field(dashboard, "reset_operation"),
            "reset_assignment": field(dashboard, "reset_assignment"),
        })
        self.assertEqual(result.status_code, 422)
        self.assertIn("Đã có nội dung duyệt được lưu", result.text)
        with connect(self.location) as db:
            self.assertEqual(db.one("SELECT COUNT(*) AS n FROM expert_slots WHERE scope = 'live'")["n"], 3)
            row = db.one("SELECT answer FROM expert_reviews WHERE scope = 'live' AND question_id = ? AND status = 'draft'", (qid,))
        self.assertEqual(row["answer"], "Nội dung đang duyệt.")
        self.assertIn("Nội dung đang duyệt.", experts[1].get(f"/review/{qid}").text)

    def test_scientific_reset_preserves_admin_adjudication(self):
        self._setup_app(12)
        self._join_experts("live")
        dashboard = self.admin.get("/admin?scope=live")
        with connect(self.location) as db:
            qid = db.one("SELECT question_id FROM expert_reviews WHERE scope = 'live' AND role = 'calibration' LIMIT 1")["question_id"]
            original = json.loads(db.one("SELECT original FROM questions WHERE id = ?", (qid,))["original"])
        history = self.admin.get(f"/admin/history/live/{qid}")
        self.assertEqual(self.admin.post(f"/admin/adjudicate/live/{qid}", data={
            "csrf": self.admin_csrf, "revision": field(history, "revision"),
            "answer": original["expected_output"], "answerability": original["additional_metadata"]["answerability"],
            "evidence_id": original["additional_metadata"]["source_chunk_ids"], "notes": "Đã chốt.",
        }).status_code, 303)
        result = self.admin.post("/admin/scientific-reset", data={
            "csrf": self.admin_csrf, "scope": "live", "reset_operation": field(dashboard, "reset_operation"),
            "reset_assignment": field(dashboard, "reset_assignment"),
        })
        self.assertEqual(result.status_code, 422)
        self.assertIn("Đã có nội dung duyệt được lưu", result.text)
        with connect(self.location) as db:
            self.assertEqual(db.one("SELECT COUNT(*) AS n FROM expert_slots WHERE scope = 'live'")["n"], 3)
            item = db.one("SELECT status, notes FROM review_items WHERE scope = 'live' AND question_id = ?", (qid,))
        self.assertEqual(item["status"], "verified")
        self.assertEqual(item["notes"], "Đã chốt.")

    def test_stratified_300_allocation_and_audit_priority(self):
        fixed_path = Path(__file__).resolve().parents[2] / "evaluation" / "eval_dataset_300_final_fixed.json"
        payload = json.loads(fixed_path.read_text(encoding="utf-8"))
        questions = [{"id": g["additional_metadata"]["query_id"], "original": g} for g in payload["goldens"]]
        allocations = allocate_scientific_batches(questions, seed=42)

        self.assertEqual(len(allocations), 300)
        batch_counts = Counter(row["batch_type"] for row in allocations)
        self.assertEqual(batch_counts, {"calibration": 15, "overlap": 30, "independent": 255})

        indep = [row for row in allocations if row["batch_type"] == "independent"]
        for slot in (1, 2, 3):
            mine = [row for row in indep if row["primary_slot"] == slot]
            self.assertEqual(len(mine), 85)
            audited = [row for row in mine if row["requires_audit"]]
            self.assertEqual(len(audited), 15)
            expected_auditor = (slot % 3) + 1
            self.assertTrue(all(row["auditor_slot"] == expected_auditor for row in audited))
            self.assertTrue(all(
                "PARTIALLY_ANSWERABLE" in row["stratum"] or "TABLE_REASONING" in row["stratum"]
                for row in audited
            ))

        self.assertEqual(Counter(row["stratum"] for row in indep if row["primary_slot"] == 1),
                         Counter(row["stratum"] for row in indep if row["primary_slot"] == 2))
        self.assertEqual(Counter(row["stratum"] for row in indep if row["primary_slot"] == 1),
                         Counter(row["stratum"] for row in indep if row["primary_slot"] == 3))

        both_indep = [
            row for row in indep
            if row["stratum"].startswith("PARTIALLY_ANSWERABLE|TABLE_REASONING|")
        ]
        self.assertEqual(len(both_indep), 15)
        self.assertTrue(all(row["requires_audit"] == 1 for row in both_indep))

    def test_consensus_with_two_legacy_submissions_and_one_new_submission(self):
        self._setup_app(12)
        experts = self._join_experts("live")
        with connect(self.location, write=True) as db:
            allocation = db.one("SELECT question_id FROM question_allocations WHERE batch_type = 'calibration' LIMIT 1")
            qid = allocation["question_id"]
            original = json.loads(db.one("SELECT original FROM questions WHERE id = ?", (qid,))["original"])
            answer, label, sources = review_content(original)
            selected = [source["CHUNK_ID"] for source in sources] if label != "UNANSWERABLE" else []
            db.execute(
                "UPDATE expert_reviews SET status = 'submitted', revision = 1, answer = ?, answerability = ?, "
                "evidence_ids = ?, review_data = '{}' WHERE question_id = ? AND reviewer_id IN "
                "(SELECT reviewer_id FROM expert_slots WHERE slot IN (1, 2) AND scope = 'live')",
                (remove_citations(answer), label, json.dumps(selected), qid),
            )
        page = experts[3].get(f"/review/{qid}")
        result = experts[3].post(f"/review/{qid}", data={
            "csrf": csrf(page), "revision": field(page, "revision"), "action": "confirm",
            "answer": remove_citations(answer), "answerability": label, "evidence_id": selected, "notes": "",
        })
        self.assertEqual(result.status_code, 303, result.text)
        with connect(self.location) as db:
            item = db.one("SELECT status FROM review_items WHERE scope = 'live' AND question_id = ?", (qid,))
        self.assertEqual(item["status"], "verified")
        [golden] = self.admin.get("/admin/export/live").json()["goldens"]
        self.assertEqual(golden["additional_metadata"]["status"], "APPROVED")

    def test_calibration_and_overlap_fleiss_kappa_and_adjudication(self):
        self._setup_app(12)
        initial = self.admin.get("/admin?scope=live")
        self.assertNotIn("Số câu giao cho người này", initial.text)
        self.assertNotIn('action="/admin/invite"', initial.text)
        self.assertIn("Khởi tạo sẽ tạo 3 tài khoản chuyên gia", initial.text)
        self.assertIn("Cần đối chiếu", initial.text)
        self.assertIn("Số câu giao cho người này", self.admin.get("/admin?scope=test").text)
        experts = self._join_experts("live")
        duplicate = self.admin.post("/admin/scientific-setup", data={
            "csrf": self.admin_csrf, "scope": "live", "setup_operation": secrets.token_hex(16),
        })
        self.assertEqual(duplicate.status_code, 422)
        with connect(self.location) as db:
            self.assertEqual(db.one("SELECT COUNT(*) AS total FROM expert_slots WHERE scope = 'live'")["total"], 3)
        self.assertNotIn("Thu hồi câu chưa bắt đầu", self.admin.get("/admin?scope=live").text)

        # Manual invite must be blocked once scientific mode is initialized
        op = field(self.admin.get("/admin?scope=live"), "operation")
        blocked = self.admin.post("/admin/invite", data={"csrf": self.admin_csrf, "scope": "live", "count": 1, "operation": op})
        self.assertEqual(blocked.status_code, 422)

        with connect(self.location) as db:
            calib = [r["question_id"] for r in db.all("SELECT question_id FROM question_allocations WHERE batch_type = 'calibration'")]
            overlap = [r["question_id"] for r in db.all("SELECT question_id FROM question_allocations WHERE batch_type = 'overlap'")]
            self.assertEqual(len(calib), 1)
            self.assertEqual(len(overlap), 2)

        # Calibration: all 3 agree -> auto-verified
        qid_c = calib[0]
        for slot in (1, 2, 3):
            page = experts[slot].get(f"/review/{qid_c}")
            chunk_id = re.search(r'name="evidence_id" type="checkbox" value="([^"]+)"', page.text).group(1)
            resp = experts[slot].post(f"/review/{qid_c}", data={
                "csrf": csrf(page), "revision": field(page, "revision"), "action": "confirm",
                "answer": "Thống nhất từ calibration", "answerability": "ANSWERABLE",
                "evidence_id": chunk_id, "notes": "", "answer_reason": "Làm rõ nội dung.",
            })
            self.assertEqual(resp.status_code, 303)

        with connect(self.location) as db:
            self.assertEqual(db.one("SELECT status FROM review_items WHERE scope = 'live' AND question_id = ?", (qid_c,))["status"], "verified")

        # Overlap: 1 item agrees, 1 item disagrees -> kappa < 1.0 & flagged for adjudication
        for idx, qid_o in enumerate(overlap):
            for slot in (1, 2, 3):
                page = experts[slot].get(f"/review/{qid_o}")
                chunk_id = re.search(r'name="evidence_id" type="checkbox" value="([^"]+)"', page.text).group(1)
                label = "PARTIALLY_ANSWERABLE" if (idx == 1 and slot in (1, 2)) else "ANSWERABLE"
                resp = experts[slot].post(f"/review/{qid_o}", data={
                    "csrf": csrf(page), "revision": field(page, "revision"), "action": "confirm",
                    "answer": f"Đáp án overlap {qid_o}", "answerability": label,
                    "evidence_id": chunk_id, "notes": "", "answer_reason": "Làm rõ nội dung.",
                })
                self.assertEqual(resp.status_code, 303)

        with connect(self.location) as db:
            self.assertEqual(db.one("SELECT status FROM review_items WHERE scope = 'live' AND question_id = ?", (overlap[0],))["status"], "verified")
            disagreed = db.one("SELECT status, revision FROM review_items WHERE scope = 'live' AND question_id = ?", (overlap[1],))
            self.assertEqual(disagreed["status"], "flagged")

        admin_view = self.admin.get("/admin?scope=live")
        self.assertIn("Chỉ số Fleiss' Kappa (Gói Overlap)", admin_view.text)
        self.assertIn("2 / 2", admin_view.text)

        history = self.admin.get(f"/admin/history/live/{overlap[1]}")
        self.assertIn("Đối chiếu kết quả chuyên gia", history.text)
        chunk_id_b = re.search(r'name="evidence_id" value="([^"]+)"', history.text).group(1)
        adj = self.admin.post(f"/admin/adjudicate/live/{overlap[1]}", data={
            "csrf": self.admin_csrf, "revision": disagreed["revision"],
            "answer": "Kết quả sau hội đồng thống nhất", "answerability": "PARTIALLY_ANSWERABLE",
            "evidence_id": chunk_id_b, "notes": "Chốt theo đa số", "answer_reason": "Chốt theo đa số.",
        })
        self.assertEqual(adj.status_code, 303)

        with connect(self.location) as db:
            self.assertEqual(db.one("SELECT status FROM review_items WHERE scope = 'live' AND question_id = ?", (overlap[1],))["status"], "verified")
            # Original expert_reviews in overlap remain untouched for Fleiss' Kappa
            er_labels = Counter(r["answerability"] for r in db.all(
                "SELECT answerability FROM expert_reviews WHERE scope = 'live' AND question_id = ? AND role = 'overlap'",
                (overlap[1],),
            ))
            self.assertEqual(er_labels, {"PARTIALLY_ANSWERABLE": 2, "ANSWERABLE": 1})

        # Direct Kappa unit check for perfect agreement
        kappa_perfect = compute_fleiss_kappa(
            1,
            [
                {"question_id": "q1", "status": "submitted", "answerability": "ANSWERABLE", "evidence_ids": '["c1"]'},
                {"question_id": "q1", "status": "submitted", "answerability": "ANSWERABLE", "evidence_ids": '["c1"]'},
                {"question_id": "q1", "status": "submitted", "answerability": "ANSWERABLE", "evidence_ids": '["c1"]'},
            ],
            {"q1": ["c1", "c2"]},
        )
        self.assertEqual(kappa_perfect["answerability_kappa"], 1.0)
        self.assertEqual(kappa_perfect["evidence_kappa"], 1.0)

    def test_cross_audit_reject_and_revision_loop(self):
        self._setup_app(12)
        experts = self._join_experts("live")

        with connect(self.location) as db:
            audited = db.one(
                "SELECT question_id, primary_slot, auditor_slot FROM question_allocations "
                "WHERE batch_type = 'independent' AND primary_slot = 1 AND requires_audit = 1 LIMIT 1"
            )
            self.assertIsNotNone(audited)
            qid = audited["question_id"]
            auditor_slot = audited["auditor_slot"]

        # Auditor cannot approve before primary submits
        auditor_wait = experts[auditor_slot].get(f"/review/{qid}")
        self.assertIn("Chờ người chấm chính", auditor_wait.text)
        premature = experts[auditor_slot].post(f"/review/{qid}", data={
            "csrf": csrf(auditor_wait), "revision": field(auditor_wait, "revision"),
            "action": "audit_approve", "audit_note": "",
        })
        self.assertEqual(premature.status_code, 409)

        # Expert 1 (Primary) submits
        p1 = experts[1].get(f"/review/{qid}")
        chunk_id = re.search(r'name="evidence_id" type="checkbox" value="([^"]+)"', p1.text).group(1)
        submit1 = experts[1].post(f"/review/{qid}", data={
            "csrf": csrf(p1), "revision": field(p1, "revision"), "action": "confirm",
            "answer": "Bản nháp sơ bộ của E1", "answerability": "PARTIALLY_ANSWERABLE",
            "evidence_id": chunk_id, "notes": "", "answer_reason": "Làm rõ nội dung.",
        })
        self.assertEqual(submit1.status_code, 303)
        self.assertEqual(self.admin.get("/admin/export/live").json()["goldens"], [])

        # Auditor rejects with error note
        a_page = experts[auditor_slot].get(f"/review/{qid}")
        self.assertIn("Bản nháp sơ bộ của E1", a_page.text)
        reject = experts[auditor_slot].post(f"/review/{qid}", data={
            "csrf": csrf(a_page), "revision": field(a_page, "revision"),
            "action": "audit_reject", "audit_note": "Thiếu thông tin liều dùng.",
        })
        self.assertEqual(reject.status_code, 303)

        # Expert 1 sees warning at queue and question page, then resubmits
        queue1 = experts[1].get("/review")
        self.assertIn("bị kiểm định chéo trả lại để sửa", queue1.text)
        revisit = experts[1].get(f"/review/{qid}")
        self.assertIn("Thiếu thông tin liều dùng.", revisit.text)
        submit2 = experts[1].post(f"/review/{qid}", data={
            "csrf": csrf(revisit), "revision": field(revisit, "revision"), "action": "confirm",
            "answer": "Bản đã bổ sung đủ liều dùng.", "answerability": "PARTIALLY_ANSWERABLE",
            "evidence_id": chunk_id, "notes": "Đã bổ sung.", "answer_reason": "Bổ sung liều dùng.",
        })
        self.assertEqual(submit2.status_code, 303)

        # Auditor approves the revised submission
        a_page2 = experts[auditor_slot].get(f"/review/{qid}")
        self.assertIn("Bản đã bổ sung đủ liều dùng.", a_page2.text)
        approve = experts[auditor_slot].post(f"/review/{qid}", data={
            "csrf": csrf(a_page2), "revision": field(a_page2, "revision"),
            "action": "audit_approve", "audit_note": "Đạt yêu cầu.",
        })
        self.assertEqual(approve.status_code, 303)

        exported = self.admin.get("/admin/export/live").json()["goldens"]
        self.assertEqual(len(exported), 1)
        meta = exported[0]["additional_metadata"]
        self.assertEqual(meta["query_id"], qid)
        self.assertEqual(meta["review_batch"], "independent")
        self.assertTrue(meta["audited"])
        self.assertEqual(meta["ANNOTATOR_IDS"], ["Bác sĩ Chuyên gia 1", f"Bác sĩ Chuyên gia {auditor_slot}"])


class SourceRendering(unittest.TestCase):
    def test_frequency_fractions_render_in_plain_text_and_html(self):
        frequency = (
            r"Rất thường gặp ($\ge1/10$);" "\n"
            r"Thường gặp ($\ge1/100$ đến $<1/10$);" "\n"
            r"Ít gặp ($\ge1/1.000$ đến $<1/100$)" "\n"
            r"Hiếm gặp ($\ge1/10.000$ đến $<1/1.000$);" "\n"
            r"Rất hiếm gặp ($<1/10.000$);"
        )
        for text in (frequency, "<table><tr><td>" + frequency + "</td></tr></table>"):
            with self.subTest(html=text.startswith("<table>")):
                rendered = render_chunk({"CHUNK_ID": "frequency", "TEXT": text})
                self.assertEqual(rendered["raw"], text)
                self.assertIn("≥ 1/10", rendered["html"])
                self.assertIn("≥ 1/1.000", rendered["html"])
                self.assertIn("≥ 1/10.000", rendered["html"])
                self.assertIn("&lt;1/10.000", rendered["html"])
                self.assertNotIn("$", rendered["html"])
                self.assertNotIn(r"\ge", rendered["html"])

    def test_plain_text_repairs_are_display_only_and_html_is_sanitized(self):
        page_text = "8\n\n--- TRANG 9 ---\n\nThường gặp: văn bản ($\\le4,5\\%$)."
        display = render_chunk({"CHUNK_ID": "sample", "TEXT": page_text})
        self.assertIn("Trang 9", display["html"])
        self.assertIn("≤ 4,5%", display["html"])
        self.assertEqual(display["raw"], page_text)
        source = '<table><tr><td rowspan="2" style="display:none">Nguồn</td></tr></table>'
        sanitized = render_chunk({"CHUNK_ID": "sample", "TEXT": source})["html"]
        self.assertIn('rowspan="2"', sanitized)
        self.assertNotIn("style=", sanitized)


if __name__ == "__main__":
    unittest.main()
