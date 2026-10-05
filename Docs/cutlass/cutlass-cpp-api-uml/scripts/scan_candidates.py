#!/usr/bin/env python3
"""Independent, deliberately over-inclusive CUTLASS source candidate ledger.

This module uses only the Python standard library. It does NOT import the entity
extractor, Tree-sitter, libclang, or preprocessed/compiler output. Source tokens
are partitioned at every ; { } boundary, including function bodies. Such an
interval is a review obligation, not a claim that the interval is a declaration.
One interval can map to multiple declarations (notably comma declarators).

All byte offsets are half-open offsets into the original UTF-8 file. Comments
and literal contents cannot become identifiers. Preprocessor branches are never
executed; all branches, including #if 0, remain in the ledger. Conditions point
to explicit branch candidates containing the unevaluated predicate and origin.
Macro invocations are lexical possibilities, NOT expansions or resolved calls.
Every identifier followed by '(' is retained, including unknown lowercase
external macros; known object-like macro names are retained too. Macro bodies
are scanned as well as ordinary source. Macro definition lookup intentionally
retains every spelling match, without pretending to resolve #undef or config.

The token-partition check establishes bookkeeping completeness for these lexical
units only. It cannot establish API completeness, valid C++, macro expansion
completeness, template identity, or semantic scope. In particular braces in
mutually exclusive branches cannot be assigned one semantic scope by this pass.

The ledger is immutable by default. --check compares a fresh scan to the frozen
ledger; --replace explicitly regenerates it after scanner changes. Mappings
belong in a separate artifact and must never shrink this candidate denominator.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
LIBRARY_PREFIXES = ("cutlass/", "cute/")
OUTSIDE_SCOPE = {
    "cutlass/util/packed_stride.hpp": {
        "classification": "source_outside_scope",
        "target_path": "tools/util/include/cutlass/util/packed_stride.hpp",
        "evidence": "audits/phase-0-source.md: fixed Git tree boundary audit",
    },
    "cutlass/version_extended.h": {
        "classification": "conditionally_generated",
        "target_path": None,
        "evidence": "include/cutlass/version.h: CUTLASS_VERSIONS_GENERATED guarded include; target absent from fixed source scope",
    },
    "cutlass/arch/array.h": {
        "classification": "missing_source",
        "target_path": None,
        "evidence": "audits/phase-0-source.md: target absent from entire fixed Git tree; this is not a compilation-failure verdict",
    },
    "cutlass/arch/numeric_types.h": {
        "classification": "missing_source",
        "target_path": None,
        "evidence": "audits/phase-0-source.md: target absent from entire fixed Git tree; this is not a compilation-failure verdict",
    },
}
IDENTIFIER = re.compile(rb"[A-Za-z_\x80-\xff][A-Za-z_0-9\x80-\xff]*")
NUMBER = re.compile(rb"(?:[0-9]|\.[0-9])(?:[eEpP][+-]|[A-Za-z_0-9.]|'[A-Za-z_0-9])*")
RAW_PREFIX = re.compile(rb'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(')
QUOTED_PREFIX = re.compile(rb"(?:u8|u|U|L)?([\"'])")
NON_CALL_WORDS = {"if", "for", "while", "switch", "catch", "sizeof", "alignof", "decltype", "static_assert"}


@dataclass(frozen=True)
class Token:
    start: int
    end: int
    kind: str
    text: str


def digest(data):
    return hashlib.sha256(data).hexdigest()


def lex(source):
    """Tokenize bytes; expose malformed literals/comments rather than skip them."""
    splices = list(re.finditer(rb"\\\r?\n", source))
    if not splices:
        return _lex_logical_source(source)
    chunks, positions, removed_totals = [], [], []
    cursor = removed = 0
    for splice in splices:
        chunks.append(source[cursor:splice.start()])
        positions.append(splice.start() - removed)
        removed += splice.end() - splice.start()
        removed_totals.append(removed)
        cursor = splice.end()
    chunks.append(source[cursor:])
    tokens, comments, issues = _lex_logical_source(b"".join(chunks))

    def original(offset, end=False):
        index = (bisect_left if end else bisect_right)(positions, offset) - 1
        return offset + (removed_totals[index] if index >= 0 else 0)

    tokens = [Token(original(t.start), original(t.end, True), t.kind, t.text) for t in tokens]
    comments = [(original(a), original(b, True)) for a, b in comments]
    for issue in issues:
        a, b = issue["byte_range"]
        issue["byte_range"] = [original(a), original(b, True)]
    return tokens, comments, issues


def _lex_logical_source(source):
    tokens, comments, issues = [], [], []
    n, pos = len(source), 0
    while pos < n:
        start = pos
        if source[pos:pos + 1].isspace():
            pos += 1
            continue
        if source.startswith(b"\\\n", pos) or source.startswith(b"\\\r\n", pos):
            pos += 3 if source.startswith(b"\\\r\n", pos) else 2
            continue
        if source.startswith(b"//", pos):
            pos = source.find(b"\n", pos)
            pos = n if pos < 0 else pos
            while pos < n and source[start:pos].rstrip(b"\r").endswith(b"\\"):
                nxt = source.find(b"\n", pos + 1)
                pos = n if nxt < 0 else nxt
            comments.append((start, pos))
            continue
        if source.startswith(b"/*", pos):
            end = source.find(b"*/", pos + 2)
            pos = n if end < 0 else end + 2
            comments.append((start, pos))
            if end < 0:
                issues.append({"kind": "unterminated_comment", "byte_range": [start, pos]})
            continue
        raw = RAW_PREFIX.match(source, pos)
        quoted = QUOTED_PREFIX.match(source, pos)
        if raw:
            close = b")" + raw.group(1) + b'"'
            end = source.find(close, raw.end())
            pos = n if end < 0 else end + len(close)
            if end < 0:
                issues.append({"kind": "unterminated_raw_literal", "byte_range": [start, pos]})
            kind = "literal"
        elif quoted:
            quote, pos = quoted.group(1), quoted.end()
            terminated = False
            while pos < n:
                if source[pos:pos + 1] == b"\\":
                    pos += 2
                elif source[pos:pos + 1] == quote:
                    pos += 1
                    terminated = True
                    break
                else:
                    pos += 1
            pos = min(pos, n)
            if not terminated:
                issues.append({"kind": "unterminated_quoted_literal", "byte_range": [start, pos]})
            kind = "literal"
        elif (match := IDENTIFIER.match(source, pos)):
            pos, kind = match.end(), "identifier"
        elif (match := NUMBER.match(source, pos)):
            # Keep digit separators inside numeric literals rather than treating
            # the apostrophe as a character-literal opening.
            pos, kind = match.end(), "number"
        else:
            pos, kind = pos + 1, "punctuation"
        tokens.append(Token(start, pos, kind, source[start:pos].decode("utf-8", "replace")))
    return tokens, comments, issues


class SourceScan:
    def __init__(self, path, source):
        self.path, self.source = path, source
        self.tokens, self.comments, self.issues = lex(source)
        self.lines = [0] + [m.end() for m in re.finditer(b"\n", source)]
        self.candidates, self.directives, self.regions, self.definitions = [], [], [], []
        self._ids = set()

    def add(self, kind, start, end, reason, conditions=(), **fields):
        identity = f"{self.path}\0{kind}\0{start}\0{end}\0{fields.get('role', '')}"
        cid = "cand_" + digest(identity.encode())[:24]
        if cid in self._ids:
            raise ValueError(f"Duplicate candidate identity: {identity!r}")
        self._ids.add(cid)
        raw = self.source[start:end]
        item = {
            "candidate_id": cid, "kind": kind, "path": self.path,
            "byte_range": [start, end],
            "line_range": [bisect_right(self.lines, start), bisect_right(self.lines, max(start, end - 1))],
            "reason": reason, "conditions": list(conditions),
            "raw_sha256": digest(raw), "mapping_status": "pending",
            **fields,
        }
        self.candidates.append(item)
        return item

    def raw(self, start, end):
        return self.source[start:end].decode("utf-8", "replace")

    def scan_preprocessor(self):
        """Record directive and branch boundaries without selecting a config."""
        # Comment-masked prefix accepts '#...' after a leading block comment.
        masked = bytearray(self.source)
        for begin, end in self.comments:
            masked[begin:end] = bytes(10 if x == 10 else 32 for x in self.source[begin:end])
        skip_until, cursor, stack = 0, 0, []

        def condition_ids():
            return [frame["branch"]["candidate_id"] for frame in stack]

        def add_region(start, end):
            if end > start:
                self.regions.append(self.add(
                    "preprocessor_region", start, end,
                    "contiguous source between directives, including inactive code",
                    condition_ids(), constant_false=any(f["false"] for f in stack)))

        for index, token in enumerate(self.tokens):
            if token.start < skip_until or token.text != "#":
                continue
            line_start = self.source.rfind(b"\n", 0, token.start) + 1
            if bytes(masked[line_start:token.start]).strip():
                continue
            end = self.source.find(b"\n", token.end)
            end = len(self.source) if end < 0 else end + 1
            while end < len(self.source) and self.source[token.start:end].rstrip(b"\r\n").endswith(b"\\"):
                nxt = self.source.find(b"\n", end)
                end = len(self.source) if nxt < 0 else nxt + 1
            add_region(cursor, token.start)
            dtokens = []
            j = index + 1
            while j < len(self.tokens) and self.tokens[j].start < end:
                dtokens.append(self.tokens[j])
                j += 1
            name = dtokens[0].text if dtokens else ""
            rest = dtokens[1:]
            expression = self.raw(rest[0].start, rest[-1].end) if rest else ""
            directive = self.add("preprocessor_directive", token.start, end,
                                 "literal preprocessor directive; not evaluated", condition_ids(),
                                 directive=name, expression=expression)
            directive["_tokens"] = rest
            self.directives.append(directive)
            if name in {"if", "ifdef", "ifndef"}:
                predicate = expression if name == "if" else f"defined({expression})"
                if name == "ifndef":
                    predicate = "!(" + predicate + ")"
                false = name == "if" and re.fullmatch(r"\(*\s*0\s*\)*", expression) is not None
                branch = self.add("preprocessor_branch", end, end,
                                  "branch body retained independently of active configuration", condition_ids(),
                                  role=directive["candidate_id"], directive_id=directive["candidate_id"],
                                  predicate=predicate, source_expression=expression,
                                  constant_false=false or any(f["false"] for f in stack))
                stack.append({"branch": branch, "predicates": [predicate], "false": branch["constant_false"]})
            elif name in {"elif", "else"}:
                if not stack:
                    self.issues.append({"kind": "unmatched_preprocessor_branch", "byte_range": [token.start, end]})
                else:
                    frame = stack.pop()
                    self.close_branch(frame["branch"], token.start)
                    prior = " || ".join("(" + p + ")" for p in frame["predicates"])
                    predicate = f"!({prior})" + (f" && ({expression})" if name == "elif" else "")
                    false = (name == "elif" and expression.strip() == "0") or any(f["false"] for f in stack)
                    branch = self.add("preprocessor_branch", end, end,
                                      "alternative branch; earlier branch predicates negated", condition_ids(),
                                      role=directive["candidate_id"], directive_id=directive["candidate_id"],
                                      predicate=predicate, source_expression=expression, constant_false=false)
                    stack.append({"branch": branch, "predicates": frame["predicates"] + [expression], "false": false})
            elif name == "endif":
                if stack:
                    self.close_branch(stack.pop()["branch"], token.start)
                else:
                    self.issues.append({"kind": "unmatched_endif", "byte_range": [token.start, end]})
            if name == "define" and rest:
                macro_name = rest[0]
                gap = self.source[macro_name.end:rest[1].start] if len(rest) > 1 else b""
                function_like = len(rest) > 1 and rest[1].text == "(" and not re.sub(rb"\\\r?\n", b"", gap)
                body_index, parameters = 1, []
                if function_like:
                    depth, body_index = 0, len(rest)
                    for k in range(1, len(rest)):
                        depth += rest[k].text == "("
                        depth -= rest[k].text == ")"
                        if depth == 0:
                            parameters = [t.text for t in rest[2:k]]
                            body_index = k + 1
                            break
                body = rest[body_index:]
                hints = sorted({t.text for t in body if t.text in {"operator", "namespace", "struct", "class", "enum", "using", "typedef", "{", ";", "#"}})
                definition = self.add("macro_definition", token.start, end,
                                      "macro body can affect or generate declarations; expansion remains pending",
                                      directive["conditions"], name=macro_name.text, function_like=function_like,
                                      parameter_tokens=parameters, declaration_generation_hints=hints,
                                      body_byte_range=[body[0].start, body[-1].end] if body else [end, end],
                                      directive_id=directive["candidate_id"])
                definition["_tokens"] = body
                self.definitions.append(definition)
            cursor = skip_until = end
        add_region(cursor, len(self.source))
        for frame in stack:
            self.close_branch(frame["branch"], len(self.source))
            self.issues.append({"kind": "unterminated_preprocessor_conditional", "candidate_id": frame["branch"]["candidate_id"]})

    def close_branch(self, branch, end):
        # Identity is tied to its opening directive (role), not its final span.
        start = branch["byte_range"][0]
        branch["byte_range"][1] = end
        branch["line_range"][1] = bisect_right(self.lines, max(start, end - 1))
        branch["raw_sha256"] = digest(self.source[start:end])

    def scan_includes(self, scope_paths):
        for directive in self.directives:
            if directive["directive"] not in {"include", "include_next", "import"}:
                continue
            toks = directive["_tokens"]
            spelling, form = None, "expression"
            if toks and toks[0].kind == "literal" and toks[0].text.startswith('"'):
                spelling, form = toks[0].text[1:-1], "quoted"
            elif toks and toks[0].text == "<" and any(t.text == ">" for t in toks):
                close = next(t for t in toks if t.text == ">")
                spelling, form = self.raw(toks[0].end, close.start).strip(), "angle"
            data = {"classification": "unresolved_expression", "target_path": None,
                    "evidence": "include operand is not a literal header name; macro/configuration resolution required"}
            if spelling is not None:
                target = "include/" + spelling
                relative = posixpath.normpath(str(PurePosixPath(self.path).parent / spelling))
                if target in scope_paths or (form == "quoted" and relative in scope_paths):
                    data = {"classification": "in_scope", "target_path": target if target in scope_paths else relative,
                            "evidence": "literal header resolves to an exact path in the fixed 824-file manifest"}
                elif spelling in OUTSIDE_SCOPE:
                    data = dict(OUTSIDE_SCOPE[spelling])
                elif spelling.startswith(LIBRARY_PREFIXES):
                    data = {"classification": "unresolved_library_source", "target_path": None,
                            "evidence": "library-prefixed literal absent from scope; full-tree or generation evidence needed"}
                else:
                    data = {"classification": "external", "target_path": None,
                            "evidence": "non-library header boundary; standard/CUDA/third-party headers are not missing definitions"}
            self.add("include_dependency", *directive["byte_range"],
                     "file dependency candidate, never an API-to-API relation", directive["conditions"],
                     directive_id=directive["candidate_id"], operand=directive["expression"],
                     header=spelling, include_form=form, **data)

    def scan_invocations(self, tokens, conditions, macro_index, origin, owner=None):
        # Precompute parenthesis matches in linear time; do not repeatedly walk
        # entire nested calls merely to retain their raw source spans.
        pairs, stack = {}, []
        for index, token in enumerate(tokens):
            if token.text == "(":
                stack.append(index)
            elif token.text == ")" and stack:
                pairs[stack.pop()] = index
        for index, token in enumerate(tokens):
            if token.kind != "identifier":
                continue
            known = token.text in macro_index
            call_like = index + 1 < len(tokens) and tokens[index + 1].text == "("
            if not known and (not call_like or token.text in NON_CALL_WORDS):
                continue
            closing = pairs.get(index + 1) if call_like else None
            end = tokens[closing].end if closing is not None else token.end
            self.add("macro_invocation", token.start, end,
                     "known macro spelling" if known else "unresolved callable or external macro; conservatively retained",
                     conditions, name=token.text, form="function_like" if call_like else "object_like",
                     origin=origin, owner_candidate_id=owner,
                     is_confirmed_macro=False, macro_spelling_known=known,
                     definition_candidates=macro_index.get(token.text, []),
                     resolution="lexical_spelling_only", arguments_byte_range=[tokens[index + 1].end, tokens[closing].start] if closing is not None else None,
                     balanced_parentheses=closing is not None if call_like else None)

    def scan_syntax(self, macro_index):
        starts = [t.start for t in self.tokens]
        code_token_count = partitioned_count = intervals = 0
        for region in self.regions:
            begin, end = region["byte_range"]
            lo = bisect_right(starts, begin - 1)
            hi = bisect_right(starts, end - 1)
            tokens = self.tokens[lo:hi]
            code_token_count += len(tokens)
            self.scan_invocations(tokens, region["conditions"], macro_index, "source")
            batch = []
            for index, token in enumerate(tokens):
                batch.append(token)
                if token.text in {";", "{", "}"} or index == len(tokens) - 1:
                    first, last = batch[0], batch[-1]
                    hints = [t.text for t in batch if t.text in {"namespace", "class", "struct", "enum", "union", "using", "typedef", "operator", "template"}]
                    self.add("syntax_interval", first.start, last.end,
                             "all source tokens partitioned at lexical semicolon/brace/preprocessor boundaries; semantic classification pending",
                             region["conditions"], region_id=region["candidate_id"],
                             terminator=last.text if last.text in {";", "{", "}"} else "preprocessor_or_eof_boundary",
                             token_count=len(batch), keyword_hints=sorted(set(hints)),
                             comma_byte_offsets=[t.start for t in batch if t.text == ","],
                             constant_false=region["constant_false"])
                    partitioned_count += len(batch)
                    intervals += 1
                    batch = []
        if partitioned_count != code_token_count:
            raise AssertionError("Lexical candidate partition lost source tokens")
        directive_token_count = sum(
            bisect_right(starts, d["byte_range"][1] - 1) - bisect_right(starts, d["byte_range"][0] - 1)
            for d in self.directives)
        if code_token_count + directive_token_count != len(self.tokens):
            raise AssertionError("Tokens not partitioned exactly into ordinary source or directives")
        for definition in self.definitions:
            self.scan_invocations(definition["_tokens"], definition["conditions"], macro_index,
                                  "macro_body", definition["candidate_id"])
        return {"all_tokens": len(self.tokens), "ordinary_source_tokens": code_token_count,
                "preprocessor_tokens": directive_token_count, "partitioned_source_tokens": partitioned_count,
                "syntax_intervals": intervals, "lexical_partition_complete": partitioned_count == code_token_count,
                "semantic_coverage_status": "unverified_pending_mapping"}


def scan_sources(sources, commit="fixture", scope_fingerprint=None):
    """Public test seam; sources is a mapping from repository paths to bytes."""
    scans, macro_index = [], defaultdict(list)
    for path, source in sorted(sources.items()):
        scan = SourceScan(path, source)
        scan.scan_preprocessor()
        scans.append(scan)
        for definition in scan.definitions:
            macro_index[definition["name"]].append(definition["candidate_id"])
    files, candidates = [], []
    for scan in scans:
        scan.scan_includes(set(sources))
        partition = scan.scan_syntax(macro_index)
        for item in scan.candidates:
            item.pop("_tokens", None)
        scan.candidates.sort(key=lambda c: (c["byte_range"], c["kind"], c["candidate_id"]))
        files.append({"path": scan.path, "source_sha256": digest(scan.source), "source_bytes": len(scan.source),
                      "candidate_counts": dict(sorted(Counter(c["kind"] for c in scan.candidates).items())),
                      "lexical_accounting": partition, "diagnostics": scan.issues})
        candidates.extend(scan.candidates)
    counts = Counter(c["kind"] for c in candidates)
    includes = Counter(c["classification"] for c in candidates if c["kind"] == "include_dependency")
    return {
        "schema_version": SCHEMA_VERSION, "commit": commit,
        "source_scope_fingerprint": scope_fingerprint,
        "scanner_contract": {
            "implementation": "independent Python byte lexer; no entity-extractor input",
            "byte_ranges": "UTF-8 raw bytes, half-open [start,end)",
            "condition_model": "branch candidate IDs with raw predicates; no configuration selected",
            "syntax_model": "conservative token partition, NOT a C++ declaration parser",
            "macro_model": "all defined macro spellings plus all identifier-parenthesis forms; NO expansion claims",
            "mapping_policy": "immutable raw candidates; mapping must be stored separately and account for every ID",
            "known_limits": [
                "macro token pasting, rescanning, #undef order and expansion outputs need a separate expansion pass",
                "syntax intervals can contain several declarations or non-API statements; map one-to-many or give evidence",
                "semantic scopes, overload identity, template conditions and branch joins remain unresolved",
                "token partition completeness is not semantic API coverage",
            ],
        },
        "files": files, "candidates": candidates,
        "summary": {"file_count": len(files), "candidate_count": len(candidates),
                    "candidate_counts": dict(sorted(counts.items())), "include_classifications": dict(sorted(includes.items())),
                    "diagnostic_count": sum(len(f["diagnostics"]) for f in files),
                    "mapping_status": "all_pending", "api_coverage_status": "not_claimed"},
    }


def build(root=ROOT, check=False, replace=False):
    manifest_bytes = (root / "data/scope.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    sources = {}
    for item in manifest["files"]:
        path = item["path"]
        source = (root / "snapshot" / path).read_bytes()
        if digest(source) != item["sha256"]:
            raise ValueError(f"Snapshot SHA-256 differs from fixed scope: {path}")
        if path in sources:
            raise ValueError(f"Duplicate scope path: {path}")
        sources[path] = source
    if len(sources) != manifest["file_count"]:
        raise ValueError("Scope file count mismatch")
    ledger = scan_sources(sources, manifest["commit"], digest(manifest_bytes))
    ledger["scanner_sha256"] = digest(Path(__file__).read_bytes())
    output = root / "data/candidates.json"
    payload = (json.dumps(ledger, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    if check:
        if not output.exists() or output.read_bytes() != payload:
            raise ValueError("Frozen candidate ledger differs from independent rescan")
    elif output.exists() and output.read_bytes() != payload and not replace:
        raise ValueError("Refusing to mutate frozen candidate ledger; review changes and use --replace")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    print(json.dumps({**ledger["summary"], "ledger_sha256": digest(payload), "check": check}, ensure_ascii=False))
    return ledger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    build(args.root, args.check, args.replace)
