"""
Screenplay Slugline (Scene Heading) Parser.
Deconstructs scene headings into INT/EXT, Location, and Time of Day.
Handles shooting script scene numbers and complex variations.
"""

import re
from typing import Optional, Tuple, Dict, Any


class SluglineParser:
    """Parser for standard industry screenplay scene headings."""

    # Standard INT/EXT prefix variants
    INT_EXT_PATTERN = re.compile(
        r"^(?:\d+[\w.-]*\s+)?(INT\./EXT\.|EXT\./INT\.|INT/EXT|EXT/INT|INT\.|EXT\.|I/E\.|I/E|INT|EXT)\b[.\s]*(.*)",
        re.IGNORECASE
    )

    # Standard Times of Day
    TIMES_OF_DAY = [
        "LATER", "MOMENTS LATER", "CONTINUOUS", "SAME TIME", "NIGHT", "DAY",
        "DAWN", "DUSK", "SUNSET", "SUNRISE", "EVENING", "MORNING", "AFTERNOON",
        "MIDNIGHT", "MAGIC HOUR", "FLASHBACK"
    ]

    TIME_PATTERN = re.compile(
        r"(?:[\s–—-]+)\b(" + "|".join(TIMES_OF_DAY) + r")\b[\s–—-]*(.*)$",
        re.IGNORECASE
    )

    SCENE_NUMBER_PATTERN = re.compile(r"^(\d+[\w.-]*)\b\s+(.*)$")
    SCENE_NUMBER_TRAILING_PATTERN = re.compile(r"^(.*?)\s+\b(\d+[\w.-]*)$")

    @classmethod
    def is_slugline(cls, line: str) -> bool:
        """Determines if a string is a standard screenplay scene heading."""
        stripped = line.strip()
        if not stripped:
            return False

        # Sluglines are predominantly uppercase
        letters = [c for c in stripped if c.isalpha()]
        if letters:
            upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if upper_ratio < 0.70:
                return False

        # Match INT/EXT prefix
        return bool(cls.INT_EXT_PATTERN.match(stripped))

    @classmethod
    def parse(cls, line: str, fallback_scene_number: int = 1) -> Dict[str, Any]:
        """
        Parses a scene heading line into structured components.
        Returns:
            dict with {scene_number, slugline, int_ext, location, time_of_day}
        """
        raw = line.strip()
        scene_number = fallback_scene_number

        # Check for leading scene number (e.g., '14  INT. POLICE STATION - DAY')
        lead_match = cls.SCENE_NUMBER_PATTERN.match(raw)
        if lead_match:
            try:
                scene_number = int(re.sub(r"\D", "", lead_match.group(1)) or fallback_scene_number)
                raw = lead_match.group(2).strip()
            except ValueError:
                pass

        # Check for trailing scene number (e.g., 'INT. POLICE STATION - DAY  14')
        trail_match = cls.SCENE_NUMBER_TRAILING_PATTERN.match(raw)
        if trail_match:
            try:
                candidate_num = int(re.sub(r"\D", "", trail_match.group(2)) or fallback_scene_number)
                # Verify trailing number matches standard scene range
                if candidate_num < 1000:
                    scene_number = candidate_num
                    raw = trail_match.group(1).strip()
            except ValueError:
                pass

        # Extract INT/EXT
        int_ext_match = cls.INT_EXT_PATTERN.match(raw)
        if not int_ext_match:
            return {
                "scene_number": scene_number,
                "slugline": line.strip(),
                "int_ext": "INT",
                "location": raw,
                "time_of_day": "DAY"
            }

        prefix_raw = int_ext_match.group(1).upper()
        remainder = int_ext_match.group(2).strip()

        # Normalize INT/EXT
        if "INT" in prefix_raw and "EXT" in prefix_raw:
            int_ext = "INT/EXT"
        elif "EXT" in prefix_raw:
            int_ext = "EXT"
        else:
            int_ext = "INT"

        # Extract Time of Day from remainder
        time_of_day = "DAY"
        location = remainder

        # Search for time delimiter (usually preceded by ' - ' or hyphen)
        time_match = cls.TIME_PATTERN.search(remainder)
        if time_match:
            time_of_day = time_match.group(1).upper()
            loc_end_idx = time_match.start()
            location = remainder[:loc_end_idx].strip()
        else:
            # Check if remainder ends with one of the times
            for t in cls.TIMES_OF_DAY:
                if remainder.upper().endswith(t):
                    time_of_day = t
                    location = remainder[:-len(t)].rstrip(" -–—,.")
                    break

        # Clean location of leading/trailing hyphens or punctuation
        location = re.sub(r"^[\s–—\-.,]+|[\s–—\-.,]+$", "", location).strip()
        if not location:
            location = "UNKNOWN LOCATION"

        normalized_slugline = f"{int_ext} {location} - {time_of_day}"

        return {
            "scene_number": scene_number,
            "slugline": normalized_slugline,
            "int_ext": int_ext,
            "location": location,
            "time_of_day": time_of_day
        }
