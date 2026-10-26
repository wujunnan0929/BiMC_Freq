"""Frozen-expert competition: separate group odds from within-group mixtures."""
import math

import torch
from torch import nn
from torch.nn import functional as F


EXPERTS = ('fused', 'cov', 'description')
FEATURE_DIM = 36  # 8 per expert + 3 disagreements + 3 class-count terms + 6 support terms


def support_quality(supports, image_prototypes, semantic_prototypes):
    """Three class summaries; never retain individual support features."""
    values = []
    for j, samples in enumerate(supports):
        x = F.normalize(samples.detach().float(), dim=-1)
        if len(x) == 0:
            raise ValueError('Every class needs support samples.')
        values.append(torch.stack((x.new_tensor(math.log1p(len(x)) / 5.), x.mean(0).norm(),
            F.cosine_similarity(image_prototypes[j:j+1], semantic_prototypes[j:j+1]).squeeze())))
    return torch.stack(values).detach()


def branch_probabilities(logits, calibration):
    return {name: F.softmax(logits[name] / calibration[key], dim=-1)
            for name, key in zip(EXPERTS, ('fused_temperature', 'cov_temperature', 'description_temperature'))}


def validate_fixed(value):
    if not isinstance(value, dict) or set(value) != {'alpha_old', 'alpha_novel', 'novel_bias_delta'}:
        raise ValueError('Fixed fusion must specify alpha_old/alpha_novel/novel_bias_delta.')
    for key, number in value.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError('Fixed fusion values must be finite real scalars.')
        if key.startswith('alpha') and not 0 <= number <= 1:
            raise ValueError('Fusion weights must lie in [0,1].')
    if abs(value['novel_bias_delta']) > 20:
        raise ValueError('Fixed novel bias is outside the supported finite range.')
    return dict(value)


def fixed_scores(probabilities, num_old, fixed, calibration):
    p = probabilities
    old = fixed['alpha_old'] * p['fused'][:, :num_old] + (1-fixed['alpha_old']) * p['cov'][:, :num_old]
    novel = fixed['alpha_novel'] * p['fused'][:, num_old:] + (1-fixed['alpha_novel']) * p['description'][:, num_old:]
    bias = calibration['novel_bias'] + fixed['novel_bias_delta']
    return torch.cat((old, novel * math.exp(bias)), dim=-1).clamp_min(1e-30)


def evidence_features(probabilities, num_old, quality, use_support_quality=True):
    """Only unlabeled query evidence, allowed class statistics and class counts."""
    c = probabilities['fused'].shape[1]
    if not 0 < num_old < c or quality.shape != (c, 3):
        raise ValueError('Competition features require both groups and Cx3 support summaries.')
    rows, predictions, features = len(probabilities['fused']), [], []
    for name in EXPERTS:
        p = probabilities[name].detach()
        old, novel = p[:, :num_old], p[:, num_old:]
        masses = [v.sum(-1).clamp_min(1e-30) for v in (old, novel)]
        features.append(((masses[0].log()-masses[1].log()) / 6.).clamp(-2, 2))
        for v, mass in zip((old, novel), masses):
            conditional = v / mass[:, None]
            top = conditional.topk(min(2, v.shape[1]), dim=-1).values
            entropy = -(conditional * conditional.clamp_min(1e-30).log()).sum(-1)
            entropy = entropy / max(1., math.log(v.shape[1]))
            features.extend((top[:, 0], entropy, top[:, 0] - (top[:, 1] if top.shape[1] > 1 else 0.)))
        prediction = p.argmax(-1)
        predictions.append(prediction)
        features.append((prediction < num_old).float())
    for left, right in ((0, 1), (0, 2), (1, 2)):
        features.append((predictions[left] != predictions[right]).float())
    p = probabilities['fused']
    features.extend(p.new_full((rows,), value) for value in
                    (math.log1p(num_old)/5., math.log1p(c-num_old)/5., num_old/c))
    old_best = p[:, :num_old].argmax(-1)
    novel_best = p[:, num_old:].argmax(-1) + num_old
    for index in (old_best, novel_best):
        selected = quality.to(p.device)[index].detach() if use_support_quality else p.new_zeros((rows, 3))
        features.extend(selected[:, j] for j in range(3))
    result = torch.stack(features, dim=-1).detach()
    if result.shape != (rows, FEATURE_DIM) or not torch.isfinite(result).all():
        raise ValueError('Non-finite or incompatible competition features.')
    return result


class CompetitiveFusion(nn.Module):
    def __init__(self, hidden=16, method='joint', use_support_quality=True, group_limit=2., mix_limit=.5):
        super().__init__()
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError('hidden must be positive.')
        if method not in ('group', 'joint') or not isinstance(use_support_quality, bool):
            raise ValueError('Unknown competitive fusion architecture.')
        if not math.isfinite(group_limit) or not 0 < group_limit <= 10 or not math.isfinite(mix_limit) or not 0 < mix_limit <= 1:
            raise ValueError('Competition limits are invalid.')
        self.hidden, self.method, self.use_support_quality = hidden, method, use_support_quality
        self.group_limit, self.mix_limit = float(group_limit), float(mix_limit)
        self.network = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def configuration(self):
        return {k: getattr(self, k) for k in ('hidden', 'method', 'use_support_quality', 'group_limit', 'mix_limit')}

    def forward(self, features):
        out = torch.tanh(self.network(features.detach()))
        if self.method == 'group':
            out = torch.cat((out[:, :1], out[:, 1:] * 0.), dim=-1)
        return out * out.new_tensor([self.group_limit, self.mix_limit, self.mix_limit])


def competitive_scores(model, probabilities, num_old, quality, fixed, calibration, blend=1., anchor=None):
    """Group odds and conditional expert weights have separate bounded updates.

    Zero blend returns the exact anchor object. At zero network output, frozen
    inference also returns the anchor; training keeps the live zero gradients.
    """
    anchor = fixed_scores(probabilities, num_old, fixed, calibration) if anchor is None else anchor
    if blend == 0 or num_old == anchor.shape[1]:
        return anchor
    features = evidence_features(probabilities, num_old, quality, model.use_support_quality)
    delta = blend * model(features)
    if not delta.requires_grad and not bool(torch.count_nonzero(delta)):
        return anchor
    def adjust(alpha, change):
        return alpha + (1-alpha)*change.clamp_min(0) + alpha*change.clamp_max(0)
    a_old = adjust(fixed['alpha_old'], delta[:, 1:2])
    a_new = adjust(fixed['alpha_novel'], delta[:, 2:3])
    p = probabilities
    old = a_old*p['fused'][:, :num_old] + (1-a_old)*p['cov'][:, :num_old]
    novel = a_new*p['fused'][:, num_old:] + (1-a_new)*p['description'][:, num_old:]
    # Preserve the fixed group masses while adapting the within-group rules.
    # Only delta[:,0] changes inter-group odds.
    old = old * (anchor[:, :num_old].sum(-1, keepdim=True) / old.sum(-1, keepdim=True).clamp_min(1e-30))
    novel = novel * (anchor[:, num_old:].sum(-1, keepdim=True) / novel.sum(-1, keepdim=True).clamp_min(1e-30))
    novel = novel * delta[:, :1].exp()
    return torch.cat((old, novel), dim=-1).clamp_min(1e-30)
