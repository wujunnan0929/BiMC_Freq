"""Conservative spatial-scale text attributes, without numerical dependencies.

These are *semantic scale hypotheses*, not measured Fourier bands. Only actual
content words are grounded; neither the label 'high frequency' nor the species
name is a grounding keyword. Source descriptions may still contain factual
errors. Lexical selection cannot verify a species' appearance.
"""
import re

from utils.frequency_descriptions import BANDS, normalize_name


PATTERNS = {
    'low': re.compile(r'\b(?:silhouette|outline|overall|uniform(?:ly)?|elongated|'
                      r'compact|rounded|slender|stocky|large|small|broad|long|short)\b', re.I),
    'middle': re.compile(r'\b(?:patch(?:es)?|band(?:s)?|border(?:s)?|contrast(?:ing|s)?|'
                         r'collar|cap|hood(?:ed)?|bib|mask|saddle|crescent|two[- ]toned)\b', re.I),
    'high': re.compile(r'\b(?:fine(?:ly)?|tiny|delicate|thin|scallop(?:ed|ing)?|'
                       r'speckl(?:ed|es|ing)|streak(?:ed|s|ing)|barr(?:ed|ing)|'
                       r'serrat(?:ed|ion)|mottl(?:ed|ing)|feather\s+edges|'
                       r'feather\s+tips|edging|fringes)\b', re.I),
}
NONVISUAL = re.compile(
    r'\b(?:flight\s+patterns?|flying\s+patterns?|glid(?:e|es|ing)|soar(?:s|ing)?|'
    r'behavior|behaviour|habitat|diet|migrat\w*|nocturnal|song|call|singing|'
    r'feeds?|feeding|swimming|diving|hunting)\b', re.I)
NEGATED = re.compile(r'\b(?:no|not|without|lacks?|absent)\b', re.I)


def build_frequency_manifest(payload, class_names, sentences_per_band=2):
    """Preserve explicit low/middle/high assignments; filter without image use.

    No fillers, reassignment, cross-band duplication, or image-based choice.
    The first distinct grounded static sentences up to the budget are retained.
    All rejections (including beyond-budget candidates) remain in the audit.
    Names are replaced with 'subject' for both mask extraction and retrieval.
    """
    if (isinstance(sentences_per_band,bool) or not isinstance(sentences_per_band,int)
            or sentences_per_band < 1):
        raise ValueError('sentences_per_band must be a positive integer.')
    if not isinstance(payload,dict) or not payload:
        raise ValueError('Frequency descriptions must be a nonempty class dictionary.')
    normalized = {}
    for name, groups in payload.items():
        key = normalize_name(name)
        if key in normalized:
            raise ValueError('Duplicate normalized class name: '+name)
        normalized[key] = groups
    if not class_names or len({normalize_name(n) for n in class_names}) != len(class_names):
        raise ValueError('Expected distinct nonempty class names.')
    classes = []
    for name in class_names:
        groups = normalized.get(normalize_name(name))
        if not isinstance(groups,dict) or set(groups) != set(BANDS):
            raise ValueError('Class requires low/middle/high lists: '+name)
        name_pattern = re.compile(r'\b'+r'[\s_-]+'.join(
            re.escape(w) for w in normalize_name(name).split())+r'\b',re.I)
        selected, rejected, seen = {g:[] for g in BANDS}, [], set()
        for band in BANDS:
            if not isinstance(groups[band],list):
                raise ValueError('Frequency band must be a list (possibly empty).')
            for index, original in enumerate(groups[band]):
                if not isinstance(original,str) or not original.strip():
                    raise ValueError('Frequency descriptions must be nonempty strings.')
                text = ' '.join(name_pattern.sub('subject',original.strip()).split())
                reason = None
                if NONVISUAL.search(text):
                    reason = 'nonvisual_or_temporal_content'
                elif NEGATED.search(text):
                    reason = 'negation_not_supported_by_lexical_grounding'
                elif not PATTERNS[band].search(text):
                    reason = 'no_scale_content_keyword'
                elif text.casefold() in seen:
                    reason = 'duplicate_text_across_or_within_bands'
                elif len(selected[band]) >= sentences_per_band:
                    reason = 'sentence_budget'
                item = {'text':text,'original_text':original,'source_index':index,
                        'source_band':band}
                if reason:
                    rejected.append({**item,'reason':reason})
                else:
                    selected[band].append(item)
                    seen.add(text.casefold())
        classes.append({'name':name,'groups':selected,'rejected':rejected})
    return {'schema_version':1,'group_order':list(BANDS),'classes':classes,
        'sentences_per_band':sentences_per_band,'class_name_policy':'replace_full_name_with_subject',
        'selection':'fixed_explicit_band_order_static_positive_content_unique_strings',
        'frequency_semantics':'spatial_scale_text_hypothesis_not_measured_spectral_band',
        'source_factuality_verified':False,
        'limitation':'Lexical filters do not establish factuality, exact spatial frequency, '
                     'or frequency selectivity of the grounded channels.',
        'coverage':{g:sum(bool(c['groups'][g]) for c in classes) for g in BANDS}}
