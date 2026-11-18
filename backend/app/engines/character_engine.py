"""
Character Extraction & Registry Engine.
Builds the comprehensive character registry tracking first/last appearance,
speaking presence, co-occurrence relationships, stated motivations, and actions.
Adheres strictly to the rule: 'Never assume two differently named characters are the same person without evidence.'
"""

import uuid
import re
from typing import List, Dict, Any, Set
from collections import defaultdict
from app.schemas.screenplay import SceneData, CharacterRecord


class CharacterEngine:
    """Engine for creating and maintaining the character registry."""

    # Patterns indicating explicit character goals and motivations in dialogue
    GOAL_PATTERNS = [
        re.compile(r"\b(?:I want|I need|I have to|I must|my plan is|we have to|we must)\s+([^.!?]+)", re.IGNORECASE),
        re.compile(r"\b(?:I'm trying to|I will not let|my goal is)\s+([^.!?]+)", re.IGNORECASE)
    ]

    DECISION_PATTERNS = [
        re.compile(r"\b(?:I decided|I choose|I'm going to|I'm leaving|I quit|we're done)\b", re.IGNORECASE),
        re.compile(r"\b(?:no more|from now on|I agree|deal)\b", re.IGNORECASE)
    ]

    @classmethod
    def build_registry(cls, scenes: List[SceneData]) -> List[CharacterRecord]:
        """
        Builds character records across all parsed scenes.
        Computes co-occurrence matrices, first/last appearances, and evidence quotes.
        """
        char_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "canonical_name": "",
            "aliases": set(),
            "first_appearance": None,
            "last_appearance": None,
            "total_dialogue_blocks": 0,
            "total_words": 0,
            "scenes_present": [],
            "relationships": defaultdict(int),
            "goals": [],
            "stated_motivations": [],
            "actions": [],
            "important_decisions": [],
            "character_changes": []
        })

        # Track appearances and dialogue across scenes
        for scene in scenes:
            scene_num = scene.scene_number
            page = scene.page_start
            scene_id = scene.scene_id

            # 1. Update presence
            for char_name in scene.characters_present:
                c = char_data[char_name]
                c["canonical_name"] = char_name
                c["scenes_present"].append(scene_num)

                appearance_info = {
                    "scene_number": scene_num,
                    "page_number": page,
                    "scene_id": scene_id,
                    "slugline": scene.slugline
                }
                if c["first_appearance"] is None:
                    c["first_appearance"] = appearance_info
                c["last_appearance"] = appearance_info

            # 2. Update Co-occurrence Relationships
            # For every pair of characters present in the same scene, increment relationship counter
            for i in range(len(scene.characters_present)):
                for j in range(i + 1, len(scene.characters_present)):
                    c1 = scene.characters_present[i]
                    c2 = scene.characters_present[j]
                    char_data[c1]["relationships"][c2] += 1
                    char_data[c2]["relationships"][c1] += 1

            # 3. Analyze Dialogue Blocks for Goals, Stated Motivations & Decisions
            for d in scene.dialogue:
                char_name = d.character
                c = char_data[char_name]
                c["total_dialogue_blocks"] += 1
                words = len(d.text.split())
                c["total_words"] += words

                # Check for explicit stated motivations / goals
                for pat in cls.GOAL_PATTERNS:
                    m = pat.search(d.text)
                    if m:
                        goal_str = f"Page {d.page_number} (Scene {scene_num}): \"{d.text.strip()}\""
                        if goal_str not in c["stated_motivations"] and len(c["stated_motivations"]) < 5:
                            c["stated_motivations"].append(goal_str)
                            c["goals"].append(m.group(0).strip())

                # Check for decisions
                for pat in cls.DECISION_PATTERNS:
                    if pat.search(d.text):
                        dec_str = f"Page {d.page_number} (Scene {scene_num}): \"{d.text.strip()}\""
                        if dec_str not in c["important_decisions"] and len(c["important_decisions"]) < 4:
                            c["important_decisions"].append(dec_str)

            # 4. Extract physical actions from scene descriptions where character is subject
            action_lines = scene.action_description.split("\n\n")
            for line in action_lines:
                for char_name in scene.characters_present:
                    if re.search(r"\b" + re.escape(char_name) + r"\b", line):
                        act_str = f"Scene {scene_num}: {line.strip()}"
                        if act_str not in char_data[char_name]["actions"] and len(char_data[char_name]["actions"]) < 5:
                            char_data[char_name]["actions"].append(act_str)

        # 5. Strictly resolve aliases ONLY when clear textual evidence exists
        cls._resolve_strict_aliases(char_data)

        # Build final CharacterRecord models
        registry: List[CharacterRecord] = []
        for name, data in sorted(char_data.items(), key=lambda x: x[1]["total_dialogue_blocks"], reverse=True):
            if not data["first_appearance"]:
                continue

            char_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"character:{name}"))
            registry.append(CharacterRecord(
                character_id=char_id,
                canonical_name=name,
                aliases=sorted(list(data["aliases"])),
                first_appearance=data["first_appearance"],
                last_appearance=data["last_appearance"],
                total_dialogue_blocks=data["total_dialogue_blocks"],
                total_words=data["total_words"],
                scenes_present=data["scenes_present"],
                relationships=dict(data["relationships"]),
                goals=data["goals"],
                stated_motivations=data["stated_motivations"],
                actions=data["actions"],
                important_decisions=data["important_decisions"],
                character_changes=data["character_changes"]
            ))

        return registry

    @classmethod
    def _resolve_strict_aliases(cls, char_data: Dict[str, Dict[str, Any]]):
        """
        Rule: 'Never assume two differently named characters are the same person without evidence.'
        Only associates aliases when the exact same token set is an explicit prefix/suffix
        (e.g. 'OFFICER MILLER' and 'MILLER' when 'OFFICER MILLER' only appears once or title is dropped).
        """
        names = list(char_data.keys())
        for n1 in names:
            for n2 in names:
                if n1 == n2:
                    continue
                # E.g. 'DETECTIVE GITTES' vs 'GITTES'
                titles = {"DETECTIVE", "OFFICER", "SERGEANT", "CAPTAIN", "DR.", "DOCTOR", "MR.", "MRS.", "MS."}
                words1 = set(n1.split())
                words2 = set(n2.split())
                if words1.issubset(words2) and (words2 - words1).issubset(titles):
                    # Valid evidence of title variation
                    char_data[n2]["aliases"].add(n1)
                    char_data[n1]["aliases"].add(n2)
