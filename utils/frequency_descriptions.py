"""Dataset metadata and explicit scale-text validation without ML dependencies."""

import ast
import json
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
BANDS = ('low', 'middle', 'high')
DATASET_SPECS = {
    'cub200': {'label': 'CUB-200-2011', 'count': 200,
               'source': 'description/modified_cub200_output.json',
               'output': 'description/cub200_frequency_descriptions_deepseek.json'},
    'cifar100': {'label': 'CIFAR-100', 'count': 100,
                 'source': 'description/cifar100_prompts_full.json',
                 'output': 'description/cifar100_frequency_descriptions.json'},
    'miniimagenet': {'label': 'mini-ImageNet', 'count': 100,
                     'source': 'description/modified_miniimagent_output_final.json',
                     'output': 'description/miniimagenet_frequency_descriptions.json'},
}


def dataset_slug(name):
    slug = re.sub(r'[-_\s]', '', str(name).lower())
    if slug not in DATASET_SPECS:
        raise ValueError('Frequency descriptions support cub200, cifar100 and miniimagenet; got ' + str(name))
    return slug


def normalize_name(name):
    return ' '.join(str(name).lower().replace('_', ' ').replace('-', ' ').split())


def dataset_class_names(dataset):
    """Read the dataset's literal CLASSES without importing torchvision/downloads."""
    slug = dataset_slug(dataset)
    tree = ast.parse((REPO_ROOT / 'datasets' / (slug + '.py')).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CLASSES' for t in node.targets):
            names = ast.literal_eval(node.value)
            if (not isinstance(names, list) or len(names) != DATASET_SPECS[slug]['count']
                    or not all(isinstance(n, str) and n.strip() for n in names)):
                raise ValueError('Invalid dataset CLASSES for ' + slug)
            if len({normalize_name(n) for n in names}) != len(names):
                raise ValueError('Ambiguous normalized dataset classes for ' + slug)
            return names
    raise ValueError('Missing literal CLASSES in dataset ' + slug)


def config_scalar(path, key, default=None):
    """Read a simple scalar from the repository's dataset/trainer YAML files.

    The launchers remain standard-library-only. Full YAML interpretation is
    still performed by YACS in main.py; multiline/ambiguous scalars are rejected.
    """
    matches = re.findall(r'^\s*' + re.escape(key) + r':\s*(.*?)\s*$',
                         Path(path).read_text(encoding='utf-8'), re.MULTILINE)
    if not matches:
        return default
    if len(matches) != 1:
        raise ValueError(f'Ambiguous {key} in {path}; supply an explicit --opts override.')
    value = matches[0].split(' #', 1)[0].strip()
    if value.startswith(('"', "'")):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f'Unsupported {key} scalar in {path}; use --opts.') from error
    if not isinstance(value, str) or value.startswith(('|', '>')):
        raise ValueError(f'Unsupported {key} scalar in {path}; use --opts.')
    return value


def description_path(dataset, supplied='auto', root=REPO_ROOT):
    if not supplied or str(supplied).lower() == 'auto':
        return Path(root) / DATASET_SPECS[dataset_slug(dataset)]['output']
    path = Path(supplied).expanduser()
    return path if path.is_absolute() else Path(root) / path


def generation_command(dataset, output=None):
    slug = dataset_slug(dataset)
    output = str(output or DATASET_SPECS[slug]['output'])
    return (f'python tools/generate_frequency_descriptions_deepseek.py --dataset {slug} '
            f'--output "{output}"')


def load_explicit_descriptions(path, class_names, *, exact=False, dataset=None):
    """Return candidates in requested class order; fail before text/image encoding."""
    path = Path(path)
    try:
        prompts = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError as error:
        command = generation_command(dataset, path) if dataset else 'python tools/generate_frequency_descriptions.py --dataset DATASET'
        raise FileNotFoundError(f'Explicit frequency description file was not found: {path}. Generate it first: {command}') from error
    if not isinstance(prompts, dict) or not prompts:
        raise ValueError(f'Explicit frequency descriptions must be a non-empty JSON object: {path}')
    normalized = {}
    for name, payload in prompts.items():
        key = normalize_name(name)
        if key in normalized:
            raise ValueError(f'Duplicate normalized class key in explicit descriptions: {name} ({path})')
        normalized[key] = payload
    requested = [normalize_name(name) for name in class_names]
    missing = [name for name, key in zip(class_names, requested) if key not in normalized]
    if missing:
        hint = f' Generate the {dataset_slug(dataset)} file with {generation_command(dataset, path)}.' if dataset else ''
        raise ValueError(f'Missing explicit frequency descriptions for class: {missing[0]}; '
                         f'{len(missing)}/{len(class_names)} classes missing in {path}.' + hint)
    if exact and set(normalized) != set(requested):
        raise ValueError(f'Frequency description file has unexpected classes for {dataset or "dataset"}: {path}')
    result, count = [], None
    for name, key in zip(class_names, requested):
        payload = normalized[key]
        if not isinstance(payload, dict) or set(payload) != set(BANDS):
            raise ValueError(f'Class {name} must contain exactly low, middle, and high lists ({path}).')
        for band in BANDS:
            values = payload[band]
            if (not isinstance(values, list) or not values
                    or not all(isinstance(v, str) and v.strip() for v in values)):
                raise ValueError(f'Descriptions for {name}/{band} must be a non-empty string list ({path}).')
            if count is None:
                count = len(values)
            if len(values) != count:
                raise ValueError(f'All class/band entries must have the same number of candidates; '
                                 f'{name}/{band} has {len(values)}, expected {count} ({path}).')
        result.append(payload)
    return result


def validate_frequency_inputs(cfg):
    """Runtime preflight, before loading datasets, drawing supports or loading CLIP."""
    f, d = cfg.TRAINER.BiMC.FREQUENCY, cfg.TRAINER.BiMC.DUAL_FREQUENCY
    g = cfg.TRAINER.BiMC.GUIDANCE
    needs_guidance = g.ENABLED and g.MODE != 'visual'
    if needs_guidance and g.SEMANTIC_SOURCE == 'ordinary':
        validate_ordinary_guidance_descriptions(cfg.DATASET.GPT_PATH, dataset_class_names(cfg.DATASET.NAME),
                                                g.CANDIDATES_PER_BAND)
    requires = ((needs_guidance and g.SEMANTIC_SOURCE == 'explicit')
                or (d.ENABLED and d.SEMANTIC_SOURCE == 'explicit')
                or (f.USE_EXPLICIT_DESCRIPTIONS and
                    (f.ENABLED or cfg.TRAINER.BiMC.CONSENSUS.ENABLED)))
    if not requires:
        return None
    path = description_path(cfg.DATASET.NAME, f.EXPLICIT_DESCRIPTION_PATH)
    candidates = load_explicit_descriptions(path, dataset_class_names(cfg.DATASET.NAME),
                                            exact=True, dataset=cfg.DATASET.NAME)
    if needs_guidance and g.SEMANTIC_SOURCE == 'explicit' and len(candidates[0]['low']) < g.CANDIDATES_PER_BAND:
        raise ValueError(f'GUIDANCE requires at least {g.CANDIDATES_PER_BAND} unique descriptions per class/band.')
    if needs_guidance and g.SEMANTIC_SOURCE == 'explicit':
        for prompts in candidates:
            for band in BANDS:
                if len(set(text.strip().lower() for text in prompts[band][:g.CANDIDATES_PER_BAND])) != g.CANDIDATES_PER_BAND:
                    raise ValueError('GUIDANCE descriptions must be unique within each class/band.')
    return {'path': str(path), 'classes': len(candidates), 'candidates_per_band': len(candidates[0]['low'])}


def validate_ordinary_guidance_descriptions(path, class_names, candidates_per_band=5):
    prompts = json.loads(Path(path).read_text(encoding='utf-8'))
    normalized = {normalize_name(name): values for name, values in prompts.items()}
    needed = 3 * candidates_per_band
    for name in class_names:
        values = normalized.get(normalize_name(name))
        if (not isinstance(values, list) or len(values) < needed
                or not all(isinstance(v, str) and v.strip() for v in values)
                or len(set(v.strip().lower() for v in values[:needed])) != needed):
            raise ValueError(f'Ordinary-text guidance control needs {needed} distinct descriptions per class; invalid class: {name}.')
    return {'classes': len(class_names), 'unique_prompts_per_class': needed}
