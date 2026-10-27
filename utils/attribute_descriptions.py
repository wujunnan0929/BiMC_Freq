"""Deterministic visible-attribute text selection, with explicit missing slots.

This is a conservative lexical heuristic, not attribute supervision or a
factuality check. It never reads image features, labels, or future class names.
"""

import json
from pathlib import Path
import re

from utils.frequency_descriptions import normalize_name


GROUPS = ('shape', 'part', 'color', 'pattern')
NONVISUAL = re.compile(
    r'\b(?:habitat|native to|found in|geograph\w*|pacific|atlantic|continent\w*|'
    r'hawaii|japan|america\w*|africa\w*|europe\w*|breed\w*|nest\w*|migrat\w*|'
    r'courtship|mate for life|monogam\w*|lifespan|life span|years? old|'
    r'feed\w*|diet|scaveng\w*|follow ships|threat\w*|endangered|'
    r'weigh\w*|pounds?|kilograms?|inches?|centimeters?|metres?|meters?|'
    r'known to|known for|often seen|typically found)\b|\d', re.I)
PATTERNS = {
    'shape': re.compile(r'\b(?:body|silhouette|proportion\w*|slender|stocky|robust|'
                        r'streamlined|compact|elongated|wingspan|long wings|short wings|'
                        r'long neck|short neck|overall shape|rounded body)\b', re.I),
    'part': re.compile(r'\b(?:bill|beak|tail|crest|crown|head|neck|wing|wings|leg|legs|'
                       r'feet|foot|eye|eyes|breast|belly|throat|rump|nape|ear|ears|'
                       r'claw|claws|muzzle|horn|horns|wheel|wheels|handle|handles)\b', re.I),
    'color': re.compile(r'\b(?:black|white|red|yellow|blue|green|brown|grey|gray|orange|'
                        r'purple|pink|golden|silver|pale|dark|light|color\w*|colour\w*|'
                        r'iridescen\w*|rufous|buff|chestnut)\b', re.I),
    'pattern': re.compile(r'\b(?:stripe\w*|striped|spot\w*|speckl\w*|barred|barring|'
                          r'streak\w*|scallop\w*|mottl\w*|edging|feather edges|'
                          r'feather tips|pattern\w*|texture\w*|scal\w*|patch\w*)\b', re.I),
}


def _bank(value, name):
    if not isinstance(value, dict) or not value:
        raise ValueError(name + ' must be a nonempty class-name JSON object.')
    normalized = {}
    for key, payload in value.items():
        key = normalize_name(key)
        if key in normalized:
            raise ValueError('Ambiguous normalized class key in ' + name)
        normalized[key] = payload
    return normalized


def _content(text, class_name):
    # Species names themselves often contain color or pattern words. Remove the
    # complete class name before checking whether a description supplies facts.
    tokens = normalize_name(class_name).split()
    expression = r'\b' + r'[\s_-]+'.join(re.escape(t) for t in tokens) + r'\b'
    return re.sub(expression, ' ', text, flags=re.I)


def inspect_sentence(text, class_name):
    if not isinstance(text, str) or not text.strip():
        return [], 'empty_or_non_string'
    content = _content(text.strip(), class_name)
    if NONVISUAL.search(content):
        return [], 'nonvisual_or_absolute_numeric_information'
    groups = [g for g in GROUPS if PATTERNS[g].search(content)]
    return groups, None if groups else 'no_supported_visible_attribute_keyword'


def build_attribute_manifest(ordinary, class_names, supplied=None, *, per_group=2,
                             ordinary_count=8):
    """Select unique sentences and retain missingness, in requested class order.

    ``supplied`` optionally maps each requested name to the four GROUPS lists.
    Extra names are not inspected or encoded. No slot is filled by repetition.
    """
    if (isinstance(per_group, bool) or not isinstance(per_group, int) or per_group < 1
            or isinstance(ordinary_count, bool) or not isinstance(ordinary_count, int)
            or ordinary_count < 1):
        raise ValueError('Text budgets must be positive integers.')
    names = list(class_names)
    if not names or len({normalize_name(n) for n in names}) != len(names):
        raise ValueError('Requested class names must be nonempty and distinct.')
    source = _bank(ordinary, 'ordinary description bank')
    override = _bank(supplied.get('classes', supplied) if isinstance(supplied, dict) else supplied,
                     'explicit attribute bank') if supplied is not None else None
    classes = []
    for name in names:
        key = normalize_name(name)
        if key not in source:
            raise ValueError('Missing ordinary descriptions for ' + name)
        sentences = source[key]
        if not isinstance(sentences, list) or not all(isinstance(t, str) for t in sentences):
            raise ValueError('Ordinary descriptions must be string lists: ' + name)
        accepted, rejected, seen = [], [], set()
        for index, text in enumerate(sentences):
            text = text.strip()
            groups, reason = inspect_sentence(text, name)
            duplicate = text.casefold() in seen
            seen.add(text.casefold())
            if reason or duplicate:
                rejected.append({'source_index': index, 'text': text,
                                 'reason': reason or 'duplicate_sentence'})
            else:
                accepted.append({'text': text, 'source_index': index,
                                 'matched_groups': groups, 'source': 'ordinary'})
        grouped = {g: [] for g in GROUPS}
        if override is None:
            for item in accepted:
                available = [g for g in item['matched_groups'] if len(grouped[g]) < per_group]
                if available:
                    group = min(available, key=lambda g: (len(grouped[g]), GROUPS.index(g)))
                    grouped[group].append(dict(item))
        else:
            if key not in override:
                raise ValueError('Missing explicit attributes for ' + name)
            payload = override[key]
            if not isinstance(payload, dict) or set(payload) != set(GROUPS):
                raise ValueError('Explicit attributes require exactly shape/part/color/pattern: ' + name)
            used = set()
            for group in GROUPS:
                values = payload[group]
                if not isinstance(values, list) or not all(isinstance(t, str) and t.strip() for t in values):
                    raise ValueError('Explicit attribute groups must be string lists (possibly empty).')
                for index, text in enumerate(values[:per_group]):
                    text = text.strip()
                    matches, reason = inspect_sentence(text, name)
                    if reason:
                        raise ValueError(f'Explicit {name}/{group} contains rejected content: {reason}')
                    if text.casefold() in used:
                        raise ValueError('Explicit attribute sentences must be unique across groups: ' + name)
                    used.add(text.casefold())
                    grouped[group].append({'text': text, 'source_index': index,
                                           'matched_groups': matches, 'source': 'explicit'})
        classes.append({'name': name, 'groups': grouped, 'ordinary': accepted[:ordinary_count],
                        'rejected': rejected,
                        'group_counts': {g: len(grouped[g]) for g in GROUPS},
                        'ordinary_count': min(len(accepted), ordinary_count)})
    complete = sum(all(len(c['groups'][g]) == per_group for g in GROUPS) for c in classes)
    return {'schema_version': 1, 'group_order': list(GROUPS), 'per_group': per_group,
            'ordinary_budget': ordinary_count,
            'selection': ('explicit_fixed_group_lists' if override is not None else
                          'deterministic_visible_keyword_unique_assignment'),
            'limitation': 'Lexical grouping is approximate, may overlap semantically, and does not verify factuality. '
                          'Unknown slots remain masked; no image-guided selection or fabricated fillers.',
            'classes': classes,
            'coverage': {'classes': len(classes), 'complete_attribute_classes': complete,
                         'groups_with_any': {g: sum(bool(c['groups'][g]) for c in classes) for g in GROUPS},
                         'slots_filled': {g: sum(len(c['groups'][g]) for c in classes) for g in GROUPS},
                         'ordinary_classes_with_full_budget': sum(c['ordinary_count'] == ordinary_count for c in classes)}}


def load_attribute_manifest(ordinary_path, class_names, attribute_path=None, **kwargs):
    ordinary = json.loads(Path(ordinary_path).read_text(encoding='utf-8'))
    supplied = json.loads(Path(attribute_path).read_text(encoding='utf-8')) if attribute_path else None
    return build_attribute_manifest(ordinary, class_names, supplied, **kwargs)
