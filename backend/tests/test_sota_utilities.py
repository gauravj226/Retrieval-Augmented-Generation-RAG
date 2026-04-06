import unittest
from dataclasses import dataclass, field
from pathlib import Path

from app.services.graph_memory import GraphMemoryStore
from app.services.long_term_memory import LongTermMemoryStore
from app.services.semantic_cache import SemanticAnswerCache
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


if __name__ == "__main__":
    unittest.main()
