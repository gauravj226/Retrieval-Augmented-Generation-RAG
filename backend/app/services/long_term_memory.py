import json
import re
from pathlib import Path
from typing import Dict, List


class LongTermMemoryStore:
    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, user_id: int, session_id: int) -> Path:
        return self.base_dir / f"user_{int(user_id)}_session_{int(session_id)}.json"

    def load(self, user_id: int, session_id: int) -> Dict:
        path = self._path(user_id, session_id)
        if not path.exists():
            return {"preferences": [], "facts": [], "last_updated": None}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"preferences": [], "facts": [], "last_updated": None}

    def save(self, user_id: int, session_id: int, profile: Dict) -> None:
        path = self._path(user_id, session_id)
        path.write_text(json.dumps(profile, ensure_ascii=True, indent=2), encoding="utf-8")

    def _extract_preferences(self, text: str) -> List[str]:
        lowered = (text or "").lower()
        patterns = [
            r"(?:please|always)\s+([^.!?\n]{8,120})",
            r"(?:prefer|i prefer)\s+([^.!?\n]{5,120})",
        ]
        found: List[str] = []
        for pat in patterns:
            for m in re.findall(pat, lowered):
                statement = m.strip(" .")
                if statement and statement not in found:
                    found.append(statement)
        if "bullet" in lowered and not any("bullet" in p for p in found):
            found.append("respond in bullet points")
        return found[:5]

    def _extract_facts(self, text: str) -> List[str]:
        facts = []
        for m in re.findall(r"\b(project|service|environment|deadline)\b[^.!?\n]{0,90}", (text or "").lower()):
            snippet = m.strip()
            if snippet and snippet not in facts:
                facts.append(snippet)
        return facts[:5]

    def update(self, user_id: int, session_id: int, user_message: str, assistant_message: str) -> Dict:
        profile = self.load(user_id=user_id, session_id=session_id)
        prefs = list(profile.get("preferences", []))
        facts = list(profile.get("facts", []))
        for p in self._extract_preferences(user_message) + self._extract_preferences(assistant_message):
            if p not in prefs:
                prefs.append(p)
        for f in self._extract_facts(user_message):
            if f not in facts:
                facts.append(f)
        profile = {
            "preferences": prefs[-20:],
            "facts": facts[-20:],
            "last_updated": user_message[:80],
        }
        self.save(user_id=user_id, session_id=session_id, profile=profile)
        return profile

    @staticmethod
    def to_prompt_context(profile: Dict) -> str:
        prefs = profile.get("preferences", [])[:6]
        facts = profile.get("facts", [])[:6]
        if not prefs and not facts:
            return ""
        lines = ["Long-term memory:"]
        if prefs:
            lines.append("Preferences: " + "; ".join(prefs))
        if facts:
            lines.append("Known facts: " + "; ".join(facts))
        return "\n".join(lines)
