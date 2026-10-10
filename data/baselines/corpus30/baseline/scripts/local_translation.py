"""Pinned local NLLB English-to-Korean translation adapter."""

import hashlib
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path
from threading import RLock

from evidence_selector import technical_identifiers
from grounded_answer_contract import AnswerValidationError
from retrieval_common import PROJECT_ROOT


CONFIG_PATH = PROJECT_ROOT / "config" / "translation.json"
GLOSSARY_PATH = PROJECT_ROOT / "config" / "translation_glossary.json"
PROTECTION_POLICY = "source_spans_alphabetic_markers_v6"
MARKER = re.compile(r"(?<![A-Za-z0-9_])ZXQ[A-Z]+(?![A-Za-z0-9_])")
TITLE_CASE_NAME = re.compile(
    r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?"
    r"(?:\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?){1,5}\b"
)
NAMED_DATA_RESOURCE = re.compile(
    r"\b(?P<name>[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?"
    r"(?:\s+[A-Z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)?){0,4})"
    r"\s+(?:dataset|data set|benchmark|corpus)\b"
)
PERCENT_BOUND = re.compile(
    r"\b(?P<bound>no more than|no less than|no fewer than|at most|at least|up to|"
    r"less than|fewer than|more than|greater than|below|above)\s*"
    r"(?P<quantity>[-+−]?(?:\d+(?:[,.]\d+)*|\.\d+)(?:[eE][-+−]?\d+)?)"
    r"\s*(?:%|percent\b)(?![A-Za-z0-9])",
    re.I,
)
PERCENT_BOUND_TARGETS = {
    "below": "미만", "less than": "미만", "fewer than": "미만",
    "above": "초과", "more than": "초과", "greater than": "초과",
    "at most": "이하", "up to": "이하", "no more than": "이하",
    "at least": "이상", "no less than": "이상", "no fewer than": "이상",
}
PERCENT_LITERAL = r"[-+−]?(?:\d+(?:[,.]\d+)*|\.\d+)(?:[eE][-+−]?\d+)?"
RESTORED_PERCENT_BOUND = re.compile(
    r"(?P<quantity>" + PERCENT_LITERAL + r")% (?P<bound>미만|초과|이하|이상)"
)
NAME_PREFIXES = {
    "a", "an", "the", "this", "that", "these", "those", "our", "their",
    "we", "they", "it", "using", "with", "without", "in", "on", "at",
    "for", "from", "by", "only", "however", "both", "each",
}


class TranslationFidelityError(AnswerValidationError):
    """A failed translation with bounded details for explicit debug output.

    The message stays free of source and invalid translated content. Callers
    may expose ``translation_diagnostics`` only in an opt-in debug response.
    """

    def __init__(self, message, *, stage, source=None, protected_source=None,
                 raw_output=None, restored_output=None, marker_mapping=None,
                 source_index=None):
        super().__init__(message)
        limit = 4000
        self.translation_diagnostics = {"stage": stage, "source_index": source_index}
        for key, value in (("source", source), ("protected_source", protected_source),
                           ("raw_output", raw_output), ("restored_output", restored_output)):
            if isinstance(value, str):
                self.translation_diagnostics[key] = value[:limit]
                if len(value) > limit:
                    self.translation_diagnostics[key + "_truncated"] = True
        if isinstance(source, str):
            self.translation_diagnostics["source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if marker_mapping:
            self.translation_diagnostics["marker_mapping"] = {
                key: value[:512] for key, value in list(marker_mapping.items())[:64]
            }


def load_translation_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported translation config schema")
    if config.get("provider") != "local_nllb":
        raise ValueError("Unsupported translation provider")
    for key in ("model", "revision", "source_language", "target_language"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"translation.{key} must be a non-empty string")
    if config["source_language"] != "eng_Latn":
        raise ValueError("translation source language must be eng_Latn")
    if config["target_language"] != "kor_Hang":
        raise ValueError("translation target language must be kor_Hang")
    if config.get("trust_remote_code") is not False:
        raise ValueError("translation trust_remote_code must be false")
    if config.get("use_safetensors") is not True:
        raise ValueError("translation model must use safetensors")
    if config.get("device") not in {"auto", "cpu", "cuda"}:
        raise ValueError("translation device must be auto, cpu, or cuda")
    if config.get("dtype") not in {"auto", "float16", "float32"}:
        raise ValueError("translation dtype must be auto, float16, or float32")
    for key in ("max_input_tokens", "max_output_tokens", "num_beams", "cpu_threads"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"translation.{key} must be a positive integer")
    if config.get("do_sample") is not False:
        raise ValueError("translation sampling must remain disabled")
    retry_enabled = config.get("retry_on_validation_failure", False)
    if type(retry_enabled) is not bool:
        raise ValueError("translation.retry_on_validation_failure must be boolean")
    if "retry_num_beams" in config and (
        type(config["retry_num_beams"]) is not int or not 2 <= config["retry_num_beams"] <= 4
    ):
        raise ValueError("translation.retry_num_beams must be an integer from 2 to 4")
    if retry_enabled and config.get("retry_num_beams", 2) <= config["num_beams"]:
        raise ValueError("translation retry beams must exceed primary beams")
    return config


def translation_config_fingerprint(config):
    canonical = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_translation_glossary(path=GLOSSARY_PATH):
    """Validate explicit source-to-target mappings, independent of any question."""
    glossary = json.loads(Path(path).read_text(encoding="utf-8"))
    if glossary.get("schema_version") != 1:
        raise ValueError("Unsupported translation glossary schema")
    if (glossary.get("source_language"), glossary.get("target_language")) != (
        "eng_Latn", "kor_Hang"
    ):
        raise ValueError("Translation glossary must map English to Korean")
    if glossary.get("matching") != "case_insensitive_word_boundary_longest_first":
        raise ValueError("Unsupported translation glossary matching policy")
    entries = glossary.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Translation glossary entries must be a list")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Every glossary entry must be an object")
        sources = entry.get("source")
        target = entry.get("target")
        if not isinstance(sources, list) or not sources:
            raise ValueError("Every glossary entry must have source aliases")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("Every glossary target must be non-empty text")
        # Numbers and marker syntax must not be introduced by a terminology map.
        if re.search(r"\d|ZXQ", target, re.I):
            raise ValueError("Glossary targets must not introduce numbers or markers")
        for source in sources:
            if not isinstance(source, str) or not re.fullmatch(
                r"[A-Za-z]+(?:[ -][A-Za-z]+)*", source
            ):
                raise ValueError("Glossary sources must be English word phrases")
            normalized = " ".join(source.casefold().split())
            if normalized in seen:
                raise ValueError("Duplicate translation glossary source alias")
            seen.add(normalized)
    return glossary


def _alphabetic_index(index):
    """Create deterministic ASCII markers without adding numeric quantities."""
    suffix = ""
    while True:
        index, remainder = divmod(index, 26)
        suffix = chr(ord("A") + remainder) + suffix
        if not index:
            return suffix.rjust(3, "A")


def _overlaps(span, occupied):
    return any(span[0] < end and start < span[1] for start, end in occupied)


def _normalize_source_punctuation(source):
    """Restore missing delimiter spaces between Latin words, outside URLs.

    Numeric commas, decimal punctuation and mathematical ratios are untouched.
    This changes only spacing; it does not split or discard a source clause.
    """
    links = [match.span() for match in re.finditer(
        r"\b(?:[A-Za-z][A-Za-z0-9+.-]*:\S+|www\.\S+)", source,
    )]
    parts, cursor = [], 0
    for match in re.finditer(r"(?<=[A-Za-z])[,;](?=[A-Za-z])", source):
        if not _overlaps(match.span(), links):
            parts.extend((source[cursor:match.end()], " "))
            cursor = match.end()
    parts.append(source[cursor:])
    return "".join(parts)


def _resource_name_spans(source):
    for match in NAMED_DATA_RESOURCE.finditer(source):
        words = list(re.finditer(r"\S+", match.group("name")))
        while words and words[0].group().casefold() in NAME_PREFIXES:
            words.pop(0)
        if words:
            yield match.start("name") + words[0].start(), match.end("name")


def _source_name_pattern(name):
    phrase = r"\s+".join(re.escape(part) for part in name.split())
    return re.compile(r"(?<![A-Za-z0-9_])" + phrase + r"(?![A-Za-z0-9_])")


def _declared_source_names(sentences):
    """Use only resource names explicitly declared in this supplied batch."""
    return tuple(sorted({
        source[start:end]
        for source in sentences
        for start, end in _resource_name_spans(source)
    }))


def protect_source_spans(source, glossary, *, source_names=()):
    """Mask only source spans, preferring explicit terms to inferred names.

    Title-case sequences are a conservative surface-form heuristic, not named
    entity recognition. Optional names must come from explicit declarations in
    the supplied translation batch. The marker table is local to one sentence.
    """
    if re.search(r"ZXQ[A-Za-z]*", source, re.I):
        raise ValueError("Source text collides with reserved translation markers")
    if isinstance(source_names, (str, bytes)):
        raise ValueError("Declared source names must be an iterable of names")
    source = _normalize_source_punctuation(source)
    candidates = []
    # Keep the explicit numeric anchor visible to the model. Only the relation
    # words are masked; restoration uses their exact source-derived bound and
    # removes an immediately adjacent equivalent percent literal, if present.
    # This is not inferred arithmetic or a parser for arbitrary units.
    for match in PERCENT_BOUND.finditer(source):
        bound = " ".join(match.group("bound").casefold().split())
        target = match.group("quantity") + "% " + PERCENT_BOUND_TARGETS[bound]
        candidates.append((*match.span("bound"), target))
    for entry in glossary["entries"]:
        for alias in entry["source"]:
            phrase = r"\s+".join(re.escape(part) for part in alias.split())
            pattern = re.compile(
                r"(?<![A-Za-z0-9_])" + phrase + r"(?![A-Za-z0-9_])", re.I
            )
            for match in pattern.finditer(source):
                target = entry["target"]
                # All-caps glossary phrases can also match the shared acronym
                # validator. Preserve that original spelling rather than erase
                # a source identifier or weaken the downstream check.
                if technical_identifiers(match.group()):
                    target += "(" + match.group() + ")"
                candidates.append((match.start(), match.end(), target))
    spans = []
    occupied = []
    # A longer phrase such as "robot grippers" wins over "grippers".
    for start, end, target in sorted(candidates, key=lambda item: (-(item[1] - item[0]), item[0])):
        if not _overlaps((start, end), occupied):
            spans.append((start, end, target))
            occupied.append((start, end))
    masked = list(source)
    for start, end in occupied:
        masked[start:end] = " " * (end - start)
    remaining = "".join(masked)
    name_candidates = []
    for identifier in technical_identifiers(remaining):
        pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(identifier) + r"(?![A-Za-z0-9_])")
        name_candidates.extend(match.span() for match in pattern.finditer(remaining))
    for match in TITLE_CASE_NAME.finditer(remaining):
        start, end = match.span()
        words = list(re.finditer(r"\S+", match.group()))
        while words and words[0].group().casefold() in NAME_PREFIXES:
            words.pop(0)
        if len(words) >= 2:
            start = match.start() + words[0].start()
            name_candidates.append((start, end))
    # A single title-case word may be a declared dataset or corpus name.
    # Scope this surface heuristic to explicit resource nouns rather than
    # protecting every capitalized sentence-initial common word.
    name_candidates.extend(_resource_name_spans(remaining))
    # Another sentence in the same supplied batch may explicitly declare a
    # resource name. Protect only its exact spelling where it is actually
    # present here, without inferring entities from ordinary capitalized words.
    for name in source_names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Declared source names must be non-empty text")
        name_candidates.extend(match.span() for match in _source_name_pattern(name).finditer(remaining))
    for start, end in sorted(name_candidates, key=lambda item: (-(item[1] - item[0]), item[0])):
        if not _overlaps((start, end), occupied):
            spans.append((start, end, source[start:end]))
            occupied.append((start, end))
    parts, mappings = [], {}
    cursor = 0
    for index, (start, end, target) in enumerate(sorted(spans)):
        marker = "ZXQ" + _alphabetic_index(index)
        parts.extend((source[cursor:start], marker))
        if RESTORED_PERCENT_BOUND.fullmatch(target) and end < len(source) and not source[end].isspace():
            parts.append(" ")
        mappings[marker] = target
        cursor = end
    parts.append(source[cursor:])
    return "".join(parts), mappings


def restore_source_spans(translated, mappings):
    """Reject missing, duplicated, mutated or invented protected markers."""
    if not isinstance(translated, str) or not translated.strip():
        raise ValueError("Translation returned empty protected content")
    observed = Counter(match.group() for match in MARKER.finditer(translated))
    if observed != Counter({marker: 1 for marker in mappings}):
        raise ValueError("Translation failed to preserve each protected source span")
    # The percentage stays visible during inference. Collapse only a directly
    # adjacent equivalent literal when restoring its source-bound marker. A
    # changed number or intervening relation stays intact for shared rejection.
    for marker, target in mappings.items():
        bound = RESTORED_PERCENT_BOUND.fullmatch(target)
        if bound is None:
            continue
        from grounded_generation import _canonical_number

        expected = _canonical_number(bound.group("quantity"))
        literal = r"(?P<quantity>" + PERCENT_LITERAL + r")\s*(?:%|percent\b|퍼센트)"
        before = re.compile(
            r"(?<![A-Za-z0-9_.+−-])" + literal + r"\s*" + re.escape(marker) + r"(?![A-Za-z0-9_])",
            re.I,
        )
        after = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(marker) + r"\s*" + literal + r"(?![A-Za-z0-9_%])",
            re.I,
        )

        before_match, after_match = before.search(translated), after.search(translated)
        if before_match and after_match:
            raise ValueError("Translation duplicated a quantity around its protected bound")

        def collapse(match):
            return marker if _canonical_number(match.group("quantity")) == expected else match.group()

        translated = before.sub(collapse, translated)
        translated = after.sub(collapse, translated)
    replacement_pattern = re.compile(
        MARKER.pattern + r"(?P<particle>으로|로|을|를|이|가|은|는|과|와)?(?=\s|[,.!?;:]|$)|" + MARKER.pattern
    )

    def replace(match):
        marker = MARKER.match(match.group()).group()
        target = mappings[marker]
        particle = match.groupdict().get("particle") or ""
        # A marker's Latin ending must not determine the particle attached to
        # a restored Korean noun. Keep Latin model-name particles untouched.
        head = target.split("(", 1)[0].rstrip()
        if particle and head and "가" <= head[-1] <= "힣":
            final = (ord(head[-1]) - ord("가")) % 28
            if particle in {"으로", "로"}:
                particle = "으로" if final not in {0, 8} else "로"
            else:
                pairs = {"을": ("을", "를"), "를": ("을", "를"),
                         "이": ("이", "가"), "가": ("이", "가"),
                         "은": ("은", "는"), "는": ("은", "는"),
                         "과": ("과", "와"), "와": ("과", "와")}
                particle = pairs[particle][0 if final else 1]
        return target + particle

    restored = replacement_pattern.sub(replace, translated).strip()
    if re.search(r"ZXQ[A-Za-z]*", restored, re.I):
        raise ValueError("Translation left an unrecognized protected marker")
    return restored


def _validate_translation_language(protected_source, raw_output, restored):
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", restored):
        raise ValueError("Korean translation contains untranslated CJK text")
    if re.search(r"[가-힣]", raw_output):
        return
    from grounded_generation import MAGNITUDE_NUMBER, NUMBER, _numeric_units

    quantity_unit = re.compile(
        "(?:" + NUMBER.pattern + r")\s*(?P<unit>[A-Za-z]+)\b", re.I
    )
    # Lists of Latin model names or quantities can remain language-neutral
    # fragments of a Korean answer. An English predicate copied unchanged is
    # not such a fragment, even if glossary restoration introduced Hangul.
    for text in (protected_source, raw_output):
        remaining = MARKER.sub("", text)
        remaining = MAGNITUDE_NUMBER.sub("", remaining)
        remaining = quantity_unit.sub(
            lambda match: "" if _numeric_units(match.group()) else match.group(),
            remaining,
        )
        remaining = NUMBER.sub("", remaining)
        remaining = re.sub(r"\b(?:and|or|vs|versus|the|a|an)\b", "", remaining, flags=re.I)
        if re.search(r"[A-Za-z]", remaining):
            raise ValueError("Translation returned untranslated English content")


class LocalEnglishKoreanTranslator:
    """Pinned NLLB translation with source-span protection and a bounded LRU."""

    provider_name = "local_nllb"

    def __init__(
        self, config_path=CONFIG_PATH, local_files_only=False, *,
        glossary_path=GLOSSARY_PATH, cache_size=128,
    ):
        self.config_path = Path(config_path)
        self.config = load_translation_config(self.config_path)
        self.config_fingerprint = translation_config_fingerprint(self.config)
        self.glossary_path = Path(glossary_path)
        self.glossary = load_translation_glossary(self.glossary_path)
        self.glossary_fingerprint = translation_config_fingerprint(self.glossary)
        if type(cache_size) is not int or not 0 <= cache_size <= 256:
            raise ValueError("translation cache_size must be an integer from 0 to 256")
        self.cache_capacity = cache_size
        self._cache = OrderedDict()
        self._cache_lock = RLock()
        self._cache_hits = self._cache_misses = self._cache_evictions = 0
        self._batch_duplicates = 0
        self._retry_attempts = self._retry_successes = self._retry_failures = 0
        self._last_request = {"effective_num_beams": [], "retry_attempts": 0, "retry_successes": 0}
        self.model_name = self.config["model"]
        self.local_files_only = bool(local_files_only)
        self._load()

    def _load(self):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Local translation requires torch, transformers, and sentencepiece."
            ) from error
        requested_device = self.config["device"]
        self.device = (
            "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        )
        if self.device == "auto":
            self.device = "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("translation.device=cuda but CUDA is unavailable")
        requested_dtype = self.config["dtype"]
        if requested_dtype == "auto":
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = getattr(torch, requested_dtype)
        common = {
            "revision": self.config["revision"],
            "trust_remote_code": self.config["trust_remote_code"],
            "local_files_only": self.local_files_only,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            src_lang=self.config["source_language"],
            **common,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            use_safetensors=self.config["use_safetensors"],
            **common,
        )
        self.device_fallback_reason = None
        if self.device == "cuda" and requested_device == "auto":
            free_bytes, _ = torch.cuda.mem_get_info()
            required = self.model.get_memory_footprint() * 1.35 + 256 * 1024**2
            if free_bytes < required:
                self.device = "cpu"
                self.dtype = torch.float32
                self.model = self.model.to(dtype=self.dtype)
                self.device_fallback_reason = "insufficient_free_cuda_memory"
        if self.device == "cpu":
            torch.set_num_threads(
                min(self.config["cpu_threads"], torch.get_num_threads())
            )
        self.model.to(self.device)
        self.model.eval()
        self._torch = torch

    def _translate_model_batch(self, sentences, *, num_beams=None):
        encoded = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        if encoded["input_ids"].shape[-1] > self.config["max_input_tokens"]:
            raise ValueError("Translation input exceeds the token budget; truncation is forbidden")
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        target_id = self.tokenizer.convert_tokens_to_ids(
            self.config["target_language"]
        )
        with self._torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=target_id,
                max_length=self.config["max_output_tokens"],
                num_beams=self.config["num_beams"] if num_beams is None else num_beams,
                do_sample=self.config["do_sample"],
            )
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        if eos_id is None:
            raise RuntimeError("Translation tokenizer has no terminal end token")
        for row in generated.tolist():
            content = [token for token in row if token != pad_id]
            if not content or content[-1] != eos_id:
                raise ValueError("Translation output ended before the source was completed")
        return [
            text.strip()
            for text in self.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        ]

    def _cache_key(self, source, source_names=()):
        return (
            source, self.model_name, self.config["revision"],
            translation_config_fingerprint(self.config),
            translation_config_fingerprint(self.glossary), PROTECTION_POLICY, tuple(source_names),
        )

    def _validate_source_translation(self, source, protected_source, mapping, raw_output, source_index):
        from grounded_generation import validate_translation_batch

        output = None
        stage = "marker_restoration"
        try:
            output = restore_source_spans(raw_output, mapping)
            stage = "language_validation"
            _validate_translation_language(protected_source, raw_output, output)
            stage = "source_fidelity_validation"
            validate_translation_batch([source], [output])
        except (AnswerValidationError, ValueError) as error:
            raise TranslationFidelityError(
                str(error), stage=stage, source=source, protected_source=protected_source,
                raw_output=raw_output, restored_output=output, marker_mapping=mapping,
                source_index=source_index,
            ) from error
        return output

    def translate(self, sentences):
        if not isinstance(sentences, list) or not sentences:
            raise ValueError("sentences must be a non-empty list")
        if any(not isinstance(sentence, str) or not sentence.strip() for sentence in sentences):
            raise ValueError("every translation input must be non-empty text")
        # The shared check is deliberately imported after module initialization.
        # Failed quantity, unit, identifier or qualifier checks must never enter
        # the cache, including when callers use the adapter outside the pipeline.
        with self._cache_lock:
            self._last_request = {"effective_num_beams": [], "retry_attempts": 0, "retry_successes": 0}
            translated_by_source = {}
            missing = []
            declared_names = _declared_source_names(sentences)
            name_context = {
                source: tuple(name for name in declared_names if _source_name_pattern(name).search(source))
                for source in sentences
            }
            for source in sentences:
                if source in translated_by_source or source in missing:
                    self._batch_duplicates += 1
                    continue
                key = self._cache_key(source, name_context[source])
                if self.cache_capacity and key in self._cache:
                    translated_by_source[source] = self._cache[key]
                    self._cache.move_to_end(key)
                    self._cache_hits += 1
                else:
                    missing.append(source)
                    self._cache_misses += 1
            if missing:
                protected = []
                for source in missing:
                    try:
                        protected.append(protect_source_spans(
                            source, self.glossary, source_names=name_context[source],
                        ))
                    except ValueError as error:
                        raise TranslationFidelityError(
                            str(error), stage="source_protection", source=source,
                            source_index=sentences.index(source),
                        ) from error
                try:
                    self._last_request["effective_num_beams"].append(self.config["num_beams"])
                    raw_outputs = self._translate_model_batch([item[0] for item in protected])
                except ValueError as error:
                    raise TranslationFidelityError(str(error), stage="model_batch") from error
                if not isinstance(raw_outputs, (list, tuple)) or len(raw_outputs) != len(missing):
                    raise TranslationFidelityError(
                        "Translation must return one output per source sentence", stage="model_batch_count",
                    )
                restored = []
                for source, (protected_source, mapping), raw_output in zip(missing, protected, raw_outputs):
                    try:
                        output = self._validate_source_translation(
                            source, protected_source, mapping, raw_output, sentences.index(source),
                        )
                    except TranslationFidelityError as first_error:
                        retry_beams = self.config.get("retry_num_beams", 2)
                        if not self.config.get("retry_on_validation_failure", False) or retry_beams <= self.config["num_beams"]:
                            raise
                        # Retry only this invalid sentence, once, using the same
                        # protected input and the same checks. Structural batch
                        # errors and input limits never reach this branch.
                        self._retry_attempts += 1
                        self._last_request["retry_attempts"] += 1
                        if retry_beams not in self._last_request["effective_num_beams"]:
                            self._last_request["effective_num_beams"].append(retry_beams)
                        try:
                            retry_outputs = self._translate_model_batch([protected_source], num_beams=retry_beams)
                            if not isinstance(retry_outputs, (list, tuple)) or len(retry_outputs) != 1:
                                raise TranslationFidelityError(
                                    "Translation retry must return one output", stage="retry_model_batch_count",
                                    source=source, source_index=sentences.index(source),
                                )
                            output = self._validate_source_translation(
                                source, protected_source, mapping, retry_outputs[0], sentences.index(source),
                            )
                        except (ValueError, RuntimeError) as retry_error:
                            self._retry_failures += 1
                            diagnostics = getattr(retry_error, "translation_diagnostics", None)
                            if not isinstance(diagnostics, dict):
                                diagnostics = {"stage": "retry_model_batch", "source_index": sentences.index(source)}
                                retry_error.translation_diagnostics = diagnostics
                            diagnostics.update({"retry_attempted": True, "retry_num_beams": retry_beams,
                                                "initial_failure": first_error.translation_diagnostics})
                            raise
                        self._retry_successes += 1
                        self._last_request["retry_successes"] += 1
                    restored.append(output)
                # Commit only after every newly translated sentence passes.
                for source, output in zip(missing, restored):
                    translated_by_source[source] = output
                    if self.cache_capacity:
                        self._cache[self._cache_key(source, name_context[source])] = output
                        while len(self._cache) > self.cache_capacity:
                            self._cache.popitem(last=False)
                            self._cache_evictions += 1
            return [translated_by_source[source] for source in sentences]

    def cache_info(self):
        with self._cache_lock:
            return {
                "enabled": bool(self.cache_capacity), "capacity": self.cache_capacity,
                "size": len(self._cache), "hits": self._cache_hits,
                "misses": self._cache_misses, "evictions": self._cache_evictions,
                "batch_duplicates": self._batch_duplicates,
            }

    def clear_cache(self, *, reset_stats=False):
        with self._cache_lock:
            self._cache.clear()
            if reset_stats:
                self._cache_hits = self._cache_misses = self._cache_evictions = 0
                self._batch_duplicates = 0
                self._retry_attempts = self._retry_successes = self._retry_failures = 0
                self._last_request = {"effective_num_beams": [], "retry_attempts": 0, "retry_successes": 0}

    def metadata(self):
        def config_label(path):
            try:
                return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            except ValueError:
                return str(path.resolve())

        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "revision": self.config["revision"],
            "license": self.config["license"],
            "device": self.device,
            "dtype": str(self.dtype).replace("torch.", ""),
            "device_fallback_reason": self.device_fallback_reason,
            "config_path": config_label(self.config_path),
            "config_fingerprint": translation_config_fingerprint(self.config),
            "glossary_path": config_label(self.glossary_path),
            "glossary_fingerprint": translation_config_fingerprint(self.glossary),
            "protection_policy": PROTECTION_POLICY,
            "cache": self.cache_info(),
            "validation_retry": {
                "enabled": self.config.get("retry_on_validation_failure", False),
                "num_beams": self.config.get("retry_num_beams", 2),
                "maximum_retries_per_sentence": 1,
                "attempts": self._retry_attempts, "successes": self._retry_successes,
                "failures": self._retry_failures,
                "last_request": {**self._last_request,
                                 "effective_num_beams": list(self._last_request["effective_num_beams"])},
            },
        }
