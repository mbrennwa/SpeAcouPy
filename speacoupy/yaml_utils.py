# speacoupy/yaml_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import re
import io

# Unicode chars that often sneak in from copy/paste:
WEIRD_WHITESPACE = {
	"\u00A0": "NO-BREAK SPACE (U+00A0)",
	"\u1680": "OGHAM SPACE MARK (U+1680)",
	"\u2000": "EN QUAD (U+2000)",
	"\u2001": "EM QUAD (U+2001)",
	"\u2002": "EN SPACE (U+2002)",
	"\u2003": "EM SPACE (U+2003)",
	"\u2004": "THREE-PER-EM SPACE (U+2004)",
	"\u2005": "FOUR-PER-EM SPACE (U+2005)",
	"\u2006": "SIX-PER-EM SPACE (U+2006)",
	"\u2007": "FIGURE SPACE (U+2007)",
	"\u2008": "PUNCTUATION SPACE (U+2008)",
	"\u2009": "THIN SPACE (U+2009)",
	"\u200A": "HAIR SPACE (U+200A)",
	"\u202F": "NARROW NO-BREAK SPACE (U+202F)",
	"\u205F": "MEDIUM MATHEMATICAL SPACE (U+205F)",
	"\u3000": "IDEOGRAPHIC SPACE (U+3000)",
	"\uFEFF": "BOM / ZERO-WIDTH NO-BREAK SPACE (U+FEFF)",
}

ZERO_WIDTH = {
	"\u200B": "ZERO-WIDTH SPACE (U+200B)",
	"\u200C": "ZERO-WIDTH NON-JOINER (U+200C)",
	"\u200D": "ZERO-WIDTH JOINER (U+200D)",
	"\u2060": "WORD JOINER (U+2060)",
}

TAB = "\t"  # YAML 1.2 forbids tabs for indentation

ALL_BAD = {**WEIRD_WHITESPACE, **ZERO_WIDTH}

@dataclass
class Offense:
	line: int   # 1-based
	col: int    # 1-based
	char: str
	description: str
	context: str

def _visible_snippet(line: str, col1: int, width: int = 80) -> str:
	# Build a caret indicator under the offending column
	caret_line = " " * (col1 - 1) + "^"
	# Trim long lines for display
	if len(line) > width:
		start = max(0, col1 - 30)
		end = min(len(line), start + width)
		snippet = line[start:end]
		caret = " " * (col1 - start - 1) + "^"
		return f"{snippet}\n{caret}"
	return f"{line}\n{caret_line}"

def find_yaml_offenses(text: str) -> List[Offense]:
	offenses: List[Offense] = []
	for i, raw_line in enumerate(text.splitlines(), start=1):
		# Identify any disallowed chars
		for j, ch in enumerate(raw_line, start=1):
			if ch in ALL_BAD or (ch == TAB and (len(raw_line) - len(raw_line.lstrip(TAB)) > 0)):
				desc = ALL_BAD.get(ch, "TAB in indentation (YAML forbids tabs)")
				offenses.append(
					Offense(
						line=i, col=j, char=ch, description=desc,
						context=_visible_snippet(raw_line, j)
					)
				)
	return offenses

def normalize_yaml_text(text: str) -> Tuple[str, int]:
	"""Return (normalized_text, replacements_count). Replaces:
	   - ALL_BAD -> ASCII space ' '
	   - Tabs in leading indentation -> 2 spaces per tab
	   - Strips leading BOM if present
	"""
	original = text
	# Strip a leading BOM if present (common when copying from Word)
	if text.startswith("\uFEFF"):
		text = text.lstrip("\uFEFF")
	# Replace ALL_BAD with simple spaces
	for ch in ALL_BAD.keys():
		text = text.replace(ch, " ")
	# Replace tabs in indentation (only leading)
	norm_lines = []
	for line in text.splitlines():
		leading = len(line) - len(line.lstrip(" \t"))
		prefix = line[:leading].replace("\t", "  ")  # two spaces per tab
		norm_lines.append(prefix + line[leading:])
	text = "\n".join(norm_lines)
	return text, (0 if text is original else 1)

def sanitize_yaml_text(text: str, strict: bool = False) -> str:
	"""If strict=False: auto-normalize; if strict=True: raise on offenses."""
	offenses = find_yaml_offenses(text)
	if strict and offenses:
		buf = io.StringIO()
		buf.write("Disallowed whitespace found in YAML:\n")
		for off in offenses[:50]:
			buf.write(f"  line {off.line}, col {off.col}: {off.description}\n{off.context}\n")
		if len(offenses) > 50:
			buf.write(f"...and {len(offenses)-50} more.\n")
		raise ValueError(buf.getvalue())
	if offenses:
		text, _ = normalize_yaml_text(text)
	return text
