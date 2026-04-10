import unittest
from dataclasses import dataclass, field
from pathlib import Path

from app.services.graph_memory import GraphMemoryStore
from app.services.long_term_memory import LongTermMemoryStore
from app.services.semantic_cache import SemanticAnswerCache
from app.services.bm25_index import HybridBM25Index
from app.services.sota_retrieval import contextualize_documents, route_mode_for_query

TEST_TMP_ROOT = Path(__file__).parent / ".tmp"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass
class MiniDocument:
    page_content: str
    metadata: dict = field(default_factory=dict)


class TestSotaUtilities(unittest.TestCase):
    def test_contextualize_documents_prepends_summary(self):
        docs = [
            MiniDocument(page_content="Policy section: employees may claim TOIL for weekend support.", metadata={"source": "hr.md"}),
            MiniDocument(page_content="Approval requires manager sign-off and timesheet entry.", metadata={"source": "hr.md"}),
        ]

        out = contextualize_documents(docs)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].metadata["context_summary"])
        self.assertTrue(out[0].page_content.startswith("Document context:"))
        self.assertIn("TOIL", out[0].metadata["raw"])

    def test_route_mode_for_query_prefers_sparse_for_exact_ids(self):
        mode = route_mode_for_query("Find ERROR-77AF log id and stack trace line")
        self.assertEqual(mode, "sparse")

    def test_semantic_cache_returns_similar_answer(self):
        cache = SemanticAnswerCache(similarity_threshold=0.8, ttl_seconds=120, max_entries=10)
        cache.put("How do I reset vpn password?", "Use the self-service portal.", kb_id=1, mode="quality")

        hit = cache.get("how can I reset my VPN password", kb_id=1, mode="quality")
        self.assertIsNotNone(hit)
        self.assertIn("self-service", hit[0].lower())

    def test_semantic_cache_respects_scope(self):
        cache = SemanticAnswerCache(similarity_threshold=0.75, ttl_seconds=120, max_entries=10)
        cache.put(
            "how to change the box expiry link setting",
            "Session A answer",
            kb_id=3,
            mode="fast",
            scope="u:1:s:33",
        )

        hit_same_scope = cache.get(
            "how to change box expiry setting",
            kb_id=3,
            mode="fast",
            scope="u:1:s:33",
        )
        hit_other_scope = cache.get(
            "how to change box expiry setting",
            kb_id=3,
            mode="fast",
            scope="u:1:s:34",
        )

        self.assertIsNotNone(hit_same_scope)
        self.assertIn("session a answer", hit_same_scope[0].lower())
        self.assertIsNone(hit_other_scope)

    def test_long_term_memory_persists_preferences(self):
        td = TEST_TMP_ROOT / "memory_case"
        td.mkdir(parents=True, exist_ok=True)
        store = LongTermMemoryStore(base_dir=td)
        store.update(
            user_id=7,
            session_id=42,
            user_message="Please always answer in bullet points and focus on deployment steps.",
            assistant_message="Understood. I will answer with bullet points and deployment focus.",
        )
        profile = store.load(user_id=7, session_id=42)

        self.assertIn("bullet", " ".join(profile.get("preferences", [])).lower())

    def test_graph_memory_extracts_and_expands_entities(self):
        td = TEST_TMP_ROOT / "graph_case"
        td.mkdir(parents=True, exist_ok=True)
        graph = GraphMemoryStore(base_dir=td)
        graph.index_documents(
            kb_id=1,
            docs=[
                MiniDocument(page_content="Neo4j connects ServiceA and ServiceB in payments graph.", metadata={"source": "arch.md"})
            ],
        )
        expansions = graph.expand_query(kb_id=1, query="ServiceA")

        self.assertIn("serviceb", [e.lower() for e in expansions])

    def test_bm25_index_ingest_time_search(self):
        td = TEST_TMP_ROOT / "bm25_case"
        td.mkdir(parents=True, exist_ok=True)
        idx = HybridBM25Index(base_dir=str(td))
        docs = [
            MiniDocument(
                page_content="By default, Box expiry links can be changed from shared link settings.",
                metadata={"source": "box.pdf", "doc_id": 10, "raw": "Box expiry links"},
            ),
            MiniDocument(
                page_content="Email quarantine login uses Barracuda web portal.",
                metadata={"source": "mail.pdf", "doc_id": 11, "raw": "quarantine"},
            ),
        ]
        idx.upsert_chunks(kb_id=3, docs=docs)

        hits = idx.search(kb_id=3, query="how to change box expiry link", top_k=2)
        self.assertTrue(hits)
        self.assertEqual(hits[0].metadata.get("source"), "box.pdf")

    def test_bm25_index_delete_document(self):
        td = TEST_TMP_ROOT / "bm25_delete_case"
        td.mkdir(parents=True, exist_ok=True)
        idx = HybridBM25Index(base_dir=str(td))
        docs = [
            MiniDocument(
                page_content="Map network drive U from This PC.",
                metadata={"source": "u-drive.pdf", "doc_id": 20, "raw": "Map network drive U"},
            ),
            MiniDocument(
                page_content="Map network drive Q from This PC.",
                metadata={"source": "q-drive.pdf", "doc_id": 21, "raw": "Map network drive Q"},
            ),
        ]
        idx.upsert_chunks(kb_id=5, docs=docs)
        removed = idx.remove_document(kb_id=5, doc_id=20)
        hits = idx.search(kb_id=5, query="map u drive", top_k=2)

        self.assertGreaterEqual(removed, 1)
        self.assertFalse(any(h.metadata.get("source") == "u-drive.pdf" for h in hits))


if __name__ == "__main__":
    unittest.main()
