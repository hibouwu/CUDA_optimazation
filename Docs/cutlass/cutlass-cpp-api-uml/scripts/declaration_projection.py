"""Byte-exact provenance for parsing a source projection with real macro edits."""
from bisect import bisect_right
from difflib import SequenceMatcher


class SourceProjection:
    def __init__(self, path, original, edits):
        self.path = path
        self.original = original
        self.edits = sorted(edits, key=lambda e: e["start_byte"])
        self.segments = []
        chunks = []
        source_cursor = projected_cursor = 0
        for edit in self.edits:
            start, end = edit["start_byte"], edit["end_byte"]
            if start < source_cursor or not start <= end <= len(original):
                raise ValueError("Overlapping or invalid projection edit")
            unchanged = original[source_cursor:start]
            if unchanged:
                chunks.append(unchanged)
                self.segments.append((projected_cursor, projected_cursor + len(unchanged), source_cursor, start, False))
                projected_cursor += len(unchanged)
            replacement = edit.get("parse_projection", edit["expanded"]).encode()
            chunks.append(replacement)
            self.segments.append((projected_cursor, projected_cursor + len(replacement), start, end, True))
            edit["projected_start_byte"] = projected_cursor
            edit["projected_end_byte"] = projected_cursor + len(replacement)
            projected_cursor += len(replacement)
            source_cursor = end
        if source_cursor < len(original):
            chunks.append(original[source_cursor:])
            self.segments.append((projected_cursor, projected_cursor + len(original) - source_cursor, source_cursor, len(original), False))
        self.projected = b"".join(chunks)
        self.starts = [s[0] for s in self.segments]
        self.original_line_starts = [0] + [i + 1 for i, c in enumerate(original) if c == 10]
        self.projected_line_starts = [0] + [i + 1 for i, c in enumerate(self.projected) if c == 10]

    def semantic_reader(self, parent=None):
        """Read semantic spelling by byte position, never by global replacement.

        Unchanged pieces delegate to the enclosing projection. A replacement
        has its own local alignment, so a grammar-only insertion cannot rewrite
        identical text in another declaration, or even another literal in the
        same declaration. Real macro expansions keep their expanded spelling.
        """
        parent = parent or (lambda a, b: self.original[a:b])
        replacements = {}
        for edit in self.edits:
            parse = edit.get('parse_projection', edit['expanded']).encode()
            semantic = edit.get('semantic_spelling', edit['expanded']).encode()
            replacements[edit['projected_start_byte']] = (semantic, SequenceMatcher(
                None, parse, semantic, autojunk=False).get_opcodes())

        def read(start, end):
            if start >= end:
                return b''
            pieces = []
            for a, b, c, d, changed in self.segments:
                lo, hi = max(start, a), min(end, b)
                if lo >= hi:
                    continue
                if not changed:
                    pieces.append(parent(c + lo - a, c + hi - a))
                    continue
                semantic, opcodes = replacements[a]
                left, right = lo - a, hi - a
                for tag, p, q, x, y in opcodes:
                    if tag == 'insert':
                        if left <= p < right or p == right == b - a:
                            pieces.append(semantic[x:y])
                    elif max(left, p) < min(right, q):
                        if tag == 'equal':
                            pieces.append(semantic[x + max(left, p) - p:x + min(right, q) - p])
                        elif tag == 'replace':
                            pieces.append(semantic[x:y])
            return b''.join(pieces)
        def origin_span(start, end):
            mapped = self.span(start, end)
            inherited = getattr(parent, 'origin_span', None)
            return inherited(mapped['start_byte'], mapped['end_byte']) if inherited else mapped
        read.origin_span = origin_span
        return read

    def offset(self, value, end=False):
        if value >= len(self.projected):
            return len(self.original)
        index = max(0, bisect_right(self.starts, value) - 1)
        a, b, c, d, expanded = self.segments[index]
        if value == a:
            return c
        return (d if end else c) if expanded else c + value - a

    def span(self, start, end):
        a = self.offset(start)
        b = self.offset(end, end=True)
        return {"path": self.path, "start_byte": a, "end_byte": b,
                "start_line": bisect_right(self.original_line_starts, a),
                "end_line": bisect_right(self.original_line_starts, max(a, b - 1))}

    def map_data(self, value):
        if isinstance(value, list):
            return [self.map_data(x) for x in value]
        if not isinstance(value, dict):
            return value
        result = {k: self.map_data(v) for k, v in value.items()}
        if value.get("path") == self.path and "start_byte" in value and "end_byte" in value:
            result["constraint_projection_range"] = {
                "coordinate_space": "constraint_expanded_source",
                "start_byte": value["start_byte"], "end_byte": value["end_byte"],
                "start_line": value.get("start_line"), "end_line": value.get("end_line"),
            }
            result.update(self.span(value["start_byte"], value["end_byte"]))
            if "raw" in value:
                result["expanded_raw"] = value.get("expanded_raw", value["raw"])
                result["raw"] = self.original[result["start_byte"]:result["end_byte"]].decode("utf-8", "replace")
        if "directive_line" in value and value.get('directive_path',self.path)==self.path:
            line = value["directive_line"]
            projected_byte = self.projected_line_starts[min(line - 1, len(self.projected_line_starts) - 1)]
            result["directive_line"] = bisect_right(self.original_line_starts, self.offset(projected_byte))
        return result
