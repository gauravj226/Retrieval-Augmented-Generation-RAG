import unittest

from app.services.ui_payload import build_ui_payload


class TestUiPayload(unittest.TestCase):
    def test_build_chart_payload_from_markdown_table(self):
        question = "Compare the financial data of Q1 and Q2."
        answer = """
| Department | Q1 | Q2 |
| --- | ---: | ---: |
| Sales | 100 | 130 |
| Support | 80 | 95 |
"""
        payload = build_ui_payload(question, answer)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["type"], "chart")
        self.assertEqual(payload["labels"], ["Sales", "Support"])
        self.assertEqual(payload["datasets"][0]["label"], "Q1")
        self.assertEqual(payload["datasets"][0]["data"], [100.0, 80.0])
        self.assertEqual(payload["datasets"][1]["label"], "Q2")
        self.assertEqual(payload["datasets"][1]["data"], [130.0, 95.0])

    def test_returns_none_for_non_comparison_question(self):
        payload = build_ui_payload("What is the leave policy?", "Simple text answer.")
        self.assertIsNone(payload)


if __name__ == "__main__":
    unittest.main()
