"""One configurable training-free FSCIL pipeline over a feature bank.

Every method in the paper is a configuration of ``run_episode``:

    classifier   'bimc' (text-visual fusion, BiMC eq. 1-4), 'proto' (visual
                 prototypes, ProtoNet/TEEN/FDR family) or 'zeroshot'
    ensemble     BiMC-dagger masked ensemble (shared covariance for base
                 classes, description nearest neighbour for novel classes)
    holistic     novel-prototype shrinkage; donor 'visual' (BiMC/TEEN) or
                 'text' (FDR code's first step)
    segments     segment-wise recomposition; 'keyword', 'keyword_fdr',
                 'keyword_random', 'freq', 'freq_classperm', 'freq_random',
                 'prior' (one class-agnostic channel set ranked by a base-only
                 statistic: within-class variance, its reverse, magnitude,
                 Fisher ratio, random, frequency sensitivity, EB ratio, or the
                 shot-adaptive EB rule) and 'eb' (soft channel-wise EB)
    anchor       pooled-within-class GDA on CLS features or on CLS + extra
                 blocks (patch DCT bands, augmentation/noise controls), fused
                 with the reference probabilities
    novel_bias   constant logit adjustment on novel columns (baseline)

A strength of 0 (lam, alpha, omega, kappa, novel_bias) switches the
corresponding component off exactly. ``EpisodeRunner`` caches everything
that does not depend on the configuration, so a selection grid of hundreds
of configurations shares one pass over the features.
"""

import copy
import json

import numpy as np

from . import keywords as kw
from .dctband import BlockBuilder, expand_blocks
from .freqmask import class_delta_means
from .gda import GDA
from .linalg import l2n, softmax, log_softmax, ridge_fit, within_class_centered
from .metrics import session_metrics, summarize_sessions
from .recompose import holistic, segment_step, frequency_alpha, n_donors

DEFAULTS = {
    'classifier': 'bimc',
    'beta': None,
    'lambda_t': 0.5,
    'ensemble': None,
    'holistic': {'donor': 'visual', 'lam': 0.1, 'tau': 16.0},
    'segments': None,
    'anchor': None,
    'novel_bias': 0.0,
    'ula': None,
}

ULA_DEFAULTS = {'tau': 0.04, 'lam': 0.4, 'T': 0.25, 'a': 8.0, 'tth': 0.06}


def ula_adjust(z, nb, tau, lam, T, a, tth):
    """Uncertainty-guided Logit Adjustment (Woo et al., CVPR 2026 Findings, Alg. 1).

    Only queries whose top-1 class is a base class, whose top-2 class is an
    incremental class and whose top-1/top-2 margin is below tau are changed:
    Delta = lam * clip((tau - m)/tau, 0, 1) * sigmoid(a (H_base/log C_b - tth)),
    top-1 logit -= Delta, top-2 logit += Delta. Mirrors the official
    adaptive_logit_adjust in github.com/sungwon-woo/ULA (models/savc/helper.py).
    """
    z = np.array(z, dtype=np.float64, copy=True)
    if z.shape[1] <= nb or lam == 0:
        return z
    top2 = np.argsort(-z, axis=1)[:, :2]
    rows = np.arange(len(z))
    i1, i2 = top2[:, 0], top2[:, 1]
    z1, z2 = z[rows, i1], z[rows, i2]
    m = z1 - z2
    h = np.clip((tau - m) / max(tau, 1e-6), 0.0, None)
    pb = softmax(z[:, :nb] / max(T, 1e-6), axis=1)
    H = -(pb * np.log(np.maximum(pb, 1e-12))).sum(1) / np.log(nb)
    w = 1.0 / (1.0 + np.exp(-a * (H - tth)))
    mask = (i1 < nb) & (i2 >= nb) & (m < tau)
    d = lam * h * w
    z[rows[mask], i1[mask]] -= d[mask]
    z[rows[mask], i2[mask]] += d[mask]
    return z

UNION_TYPES = ('freq_union', 'var_union', 'mag_union', 'fisher_union', 'random_union')
PRIOR_RANKS = ('var', 'lowvar', 'mag', 'fisher', 'random', 'freq', 'ebratio', 'ebmask')

SEGMENT_DEFAULTS = {'type': 'keyword', 'q': 50, 'r': 29, 'alpha': 0.2, 'tau': 22.0,
                    'sim': 'text', 'm': 8, 'min_df': 2, 'max_df': 0.3,
                    'sigmas': [1, 2, 4], 'bands': [0, 1, 2], 'alpha_bands': None,
                    'alpha_mod': None, 'seed': 0,
                    'rank': 'var', 'k': 0.15, 'scope': 'segment', 'c': 1.0, 's2': 'analytic',
                    'draws': 8}

ANCHOR_DEFAULTS = {'blocks': [], 'kappa': 0.0, 'kappa_mode': 'debias', 'omega': 0.2, 'temperature': 0.3,
                   'fusion': 'prob', 'shrink': 'lw', 'ridge': 0.0, 'block_diag': False,
                   'mean_source': 'raw', 'eb': None, 'eta': 1.0}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_config(cfg, bank):
    c = deep_merge(DEFAULTS, cfg)
    if c['beta'] is None:
        c['beta'] = float(bank.meta.get('beta', 0.5))
    if c['ensemble'] is not None:
        e = {'alpha': float(bank.meta.get('ensemble_alpha', 0.6)), 'gamma_base': 1.0,
             'gamma_inc': 5.0}
        e.update(c['ensemble'])
        c['ensemble'] = e
    if c['segments'] is not None:
        c['segments'] = deep_merge(SEGMENT_DEFAULTS, c['segments'])
    if c['anchor'] is not None:
        c['anchor'] = deep_merge(ANCHOR_DEFAULTS, c['anchor'])
    if c['ula'] is not None:
        c['ula'] = deep_merge(ULA_DEFAULTS, c['ula'])
    return c


def _key(*parts):
    return json.dumps(parts, sort_keys=True, default=str)


class EpisodeRunner:
    def __init__(self, bank, episode):
        self.bank, self.ep = bank, episode
        self.cache = {}
        ytr = bank.labels('train')
        self.ytr = ytr
        self.classes = [np.asarray(c) for c in episode.classes]
        self.all_cls = np.concatenate(self.classes)
        self.col = {int(c): i for i, c in enumerate(self.all_cls)}
        self.n_cols = [sum(len(c) for c in self.classes[:t + 1]) for t in range(len(self.classes))]
        self.base_cls = self.classes[0]
        self.nb = len(self.base_cls)
        Ftr = bank.feat('train', 'orig')
        self.sup_rows = [np.asarray(s) for s in episode.support]
        self.sup_feat = [Ftr[s].astype(np.float64) for s in self.sup_rows]
        self.sup_lab = [ytr[s] for s in self.sup_rows]
        self.raw_mean = np.zeros((len(self.all_cls), bank.dim))
        self.counts = np.zeros(len(self.all_cls))
        for t, cls in enumerate(self.classes):
            for c in cls:
                x = self.sup_feat[t][self.sup_lab[t] == c]
                if len(x) == 0:
                    raise ValueError(f'class {c} has no support in session {t}')
                self.raw_mean[self.col[int(c)]] = x.mean(0)
                self.counts[self.col[int(c)]] = len(x)
        self.raw_proto = l2n(self.raw_mean)
        self.q_rows = np.asarray(episode.queries[-1])
        self.q_lab = bank.labels(episode.query_split)[self.q_rows]
        self.Fq = bank.feat(episode.query_split, 'orig')[self.q_rows].astype(np.float64)
        order = np.argsort(self.q_rows)
        self.qpos = []
        for q in episode.queries:
            pos = order[np.searchsorted(self.q_rows[order], q)]
            if not np.array_equal(self.q_rows[pos], q):
                raise ValueError('session queries must be a subset of the last session queries')
            self.qpos.append(pos)
        self.text_prompt = bank.text_prompt.astype(np.float64)[self.all_cls]
        self.desc_proto = bank.desc_proto.astype(np.float64)[self.all_cls]

    # ------------------------------------------------------------- text side
    def text_cal(self, lam_t):
        return (1.0 - lam_t) * self.text_prompt + lam_t * self.desc_proto

    def knn_scores(self):
        k = _key('knn')
        if k not in self.cache:
            D = self.bank.desc_feat.astype(np.float32)
            tgt = self.bank.desc_target
            S = self.Fq.astype(np.float32) @ D.T
            out = np.full((len(self.Fq), len(self.all_cls)), -np.inf, dtype=np.float32)
            for c in self.all_cls:
                rows = np.flatnonzero(tgt == c)
                if len(rows):
                    out[:, self.col[int(c)]] = S[:, rows].max(1)
            self.cache[k] = out.astype(np.float64)
        return self.cache[k]

    # ------------------------------------------------------ frequency deltas
    def freq_deltas(self, sigmas):
        k = _key('delta', list(sigmas))
        if k not in self.cache:
            out = np.zeros((len(self.all_cls), len(sigmas), self.bank.dim))
            for t, cls in enumerate(self.classes):
                out[[self.col[int(c)] for c in cls]] = class_delta_means(
                    self.bank, 'train', self.sup_rows[t], self.sup_lab[t], cls, sigmas)
            common = out[:self.nb].mean(0)
            self.cache[k] = (out, common)
        return self.cache[k]

    # -------------------------------------------------------- recomposition
    def vis_cal(self, c):
        hk = _key('vis', c['holistic'], c['segments'],
                  c['lambda_t'] if self._uses_text(c) else None)
        if hk in self.cache:
            return self.cache[hk]
        P = self.raw_proto.copy()
        h = c['holistic']
        base_vis = self.raw_proto[:self.nb]
        tc = l2n(self.text_cal(c['lambda_t']))
        nov = np.arange(self.nb, len(self.all_cls))
        if h is not None and h.get('lam', 0) and len(nov):
            if h['donor'] == 'visual':
                src_n, src_b = P[nov], base_vis
            elif h['donor'] == 'text':
                src_n, src_b = tc[nov], tc[:self.nb]
            else:
                raise ValueError('holistic donor must be visual or text')
            P[nov] = holistic(P[nov], base_vis, src_n, src_b, h['lam'], h['tau'])
        s = c['segments']
        stats = None
        if s is not None and len(nov):
            self._cur_lam_t = c['lambda_t']
            P[nov] = self._segments(P, nov, s, tc, base_vis)
            stats = {'channels_touched': float(np.mean(self._seg_cov)) if self._seg_cov else 0.0,
                     'segments_per_class': float(np.mean(self._seg_n)) if self._seg_n else 0.0}
        self.cache[hk] = P
        self.cache['segstats' + hk] = stats
        return P

    def _uses_text(self, c):
        h, s = c['holistic'], c['segments']
        return bool((h and h.get('donor') == 'text') or (s and s.get('sim') == 'text'))

    def _eb_stats(self, s, tc, base_vis):
        """Base-only statistics shared by the EB variants.

        sig2: pooled within-class variance per channel (raw features);
        g2:   mean squared norm of base class means;
        t2:   donor error variance per channel, leave-one-out on base classes
              (text- or visual-similarity weighted top-r donors, as for novel
              classes), corrected for the base prototypes' own sampling noise.
        """
        nb = self.nb
        ek = _key('ebstats', s['sim'], s['tau'], s['r'], getattr(self, '_cur_lam_t', None))
        if ek not in self.cache:
            Xc, _ = within_class_centered(self.sup_feat[0], self.sup_lab[0], self.base_cls)
            sig2 = (Xc ** 2).sum(0) / max(len(Xc) - nb, 1)
            g2 = float((np.linalg.norm(self.raw_mean[:nb], axis=1) ** 2).mean())
            src = tc if s['sim'] == 'text' else self.raw_proto
            S = l2n(src[:nb]) @ l2n(src[:nb]).T
            np.fill_diagonal(S, -np.inf)
            r = min(n_donors(s['r'], nb), nb - 1)
            D = np.zeros_like(base_vis)
            for b in range(nb):
                top = np.argpartition(-S[b], r - 1)[:r]
                D[b] = l2n(softmax(s['tau'] * S[b, top]) @ base_vis[top])
            E2 = ((base_vis - D) ** 2).mean(0)
            noise_b = sig2 / (self.counts[:nb].mean() * g2)
            t2 = np.maximum(E2 - noise_b, 1e-3 * E2.mean())
            self.cache[ek] = (sig2, g2, t2, src)
        return self.cache[ek]

    def _shot_var(self, K, mode, draws, sig2, g2):
        """s_j^2(K): per-channel squared error of a K-shot unit prototype.

        'analytic' = sig2 / (K g2). 'empirical' measures it directly on the base
        classes: draw K of a base class's training images, normalise their mean
        and compare with the normalised mean of all its images (fixed seed).
        """
        K = int(K)
        if mode == 'analytic':
            return sig2 / (max(K, 1) * g2)
        if mode != 'empirical':
            raise ValueError('s2 must be analytic or empirical')
        k = _key('shotvar', K, int(draws))
        if k not in self.cache:
            X, y = self.sup_feat[0], self.sup_lab[0]
            rng = np.random.RandomState(9173 + K)
            acc, n = np.zeros(X.shape[1]), 0
            for b, c in enumerate(self.base_cls):
                Xb = X[y == c]
                if len(Xb) <= K:
                    continue
                ref = self.raw_proto[b]
                for _ in range(int(draws)):
                    pick = rng.choice(len(Xb), K, replace=False)
                    acc += (l2n(Xb[pick].mean(0)) - ref) ** 2
                    n += 1
            if n == 0:
                raise ValueError(f'no base class has more than {K} images')
            self.cache[k] = acc / n
        return self.cache[k]

    def _eb(self, P, nov, s, tc, base_vis):
        """Channel-wise empirical-Bayes recomposition (shot-aware, closed form).

        Each novel prototype p_hat (K shots) is pulled toward a base-donor estimate
        d (text-similarity weighted base prototypes, as in K0) channel by channel:
            p_j <- p_hat_j + rho_j (d_j - p_hat_j),
            rho_j = lam * s_j^2 / (s_j^2 + t_j^2),
        s_j^2: sampling variance of a K-shot unit prototype (``_shot_var``),
        t_j^2: donor error variance (``_eb_stats``). Few shots or noisy channels
        borrow more; rho does not need a per-shot strength. eb_uniform replaces
        rho by its per-class mean (= holistic shrinkage); eb_shuffled permutes rho
        across channels (same distribution, wrong channels).
        """
        nb = self.nb
        sig2, g2, t2, src = self._eb_stats(s, tc, base_vis)
        Sn = l2n(src[nov]) @ l2n(src[:nb]).T
        r = n_donors(s['r'], nb)
        out = np.zeros((len(nov), base_vis.shape[1]))
        perm = np.random.RandomState(4242 + int(s['seed'])).permutation(base_vis.shape[1])
        rhos = []
        for i, j in enumerate(nov):
            top = np.argpartition(-Sn[i], r - 1)[:r]
            d = l2n(softmax(s['tau'] * Sn[i, top]) @ base_vis[top])
            s2 = self._shot_var(self.counts[j], s['s2'], s['draws'], sig2, g2)
            rho = np.clip(s['alpha'] * s2 / (s2 + t2), 0.0, 1.0)
            if s['type'] == 'eb_uniform':
                rho = np.full_like(rho, rho.mean())
            elif s['type'] == 'eb_shuffled':
                rho = rho[perm]
            rhos.append(rho)
            out[i] = l2n(P[j] + rho * (d - P[j]))
        R = np.asarray(rhos)
        self._seg_cov = [float(x) for x in (R.mean(1) * base_vis.shape[1])]
        self._seg_n = [1] * len(nov)
        return out

    def _prior_mask(self, s, K, tc, base_vis):
        """One channel set shared by all novel classes, from base data only.

        rank   var     top-k pooled within-class variance (where a K-shot mean errs most)
               lowvar  bottom-k (direction control)
               mag     top-k mean |activation|;  fisher  top-k between/within ratio
               random  k random channels;  freq  top-k blur sensitivity (base average)
               ebratio top-k of s_j^2 / t_j^2 (EB noise-to-donor-error ratio)
               ebmask  every channel with s_j^2(K) > c * t_j^2: borrow where the
                       donor is expected to beat the K-shot mean (no k; shrinks
                       as K grows)
        A float k <= 1 is a fraction of the dimension (1.0 = every channel); an int is a count.
        """
        rank = s['rank']
        if rank not in PRIOR_RANKS:
            raise ValueError('unknown prior rank ' + str(rank))
        D = self.bank.dim
        k = int(round(s['k'] * D)) if isinstance(s['k'], float) and s['k'] <= 1.0 else int(s['k'])
        k = max(1, min(k, D))
        ck = _key('pmask', rank, k, s['seed'], s['sigmas'], s['bands'], s['c'], s['s2'], s['draws'],
                  int(K) if rank == 'ebmask' else None,
                  (s['sim'], s['tau'], s['r'], getattr(self, '_cur_lam_t', None))
                  if rank.startswith('eb') else None)
        if ck in self.cache:
            return self.cache[ck]
        X, y = self.sup_feat[0], self.sup_lab[0]
        m = np.zeros(D, dtype=bool)
        if rank in ('var', 'lowvar', 'fisher'):
            Xc, _ = within_class_centered(X, y, self.base_cls)
            wvar = (Xc ** 2).mean(0)
            if rank == 'var':
                m = kw.topq_mask(wvar, k)
            elif rank == 'lowvar':
                m = kw.topk_signed(-wvar, k)
            else:
                m = kw.topq_mask(self.raw_mean[:self.nb].var(0) / np.maximum(wvar, 1e-12), k)
        elif rank == 'mag':
            m = kw.topq_mask(np.abs(X).mean(0), k)
        elif rank == 'random':
            m[np.random.RandomState(12345 + int(s['seed'])).choice(D, k, replace=False)] = True
        elif rank == 'freq':
            deltas, common = self.freq_deltas(s['sigmas'])
            A = np.stack([np.abs(deltas[:self.nb, b] - common[b]).mean(0) for b in s['bands']])
            m = kw.topq_mask((A / np.maximum(A.mean(1, keepdims=True), 1e-12)).max(0), k)
        else:
            sig2, g2, t2, _ = self._eb_stats(s, tc, base_vis)
            if rank == 'ebratio':
                m = kw.topq_mask(sig2 / t2, k)
            else:
                s2 = self._shot_var(K, s['s2'], s['draws'], sig2, g2)
                m = s2 > float(s['c']) * t2
        self.cache[ck] = m
        return m

    def _segments(self, P, nov, s, tc, base_vis):
        out = P[nov].copy()
        self._seg_cov, self._seg_n = [], []
        typ = s['type']
        if typ in ('eb', 'eb_uniform', 'eb_shuffled'):
            return self._eb(P, nov, s, tc, base_vis)
        lib = rand_lib = None
        if typ.startswith('keyword'):
            coord = 'fdr' if typ == 'keyword_fdr' else 'final'
            lk = _key('lib', coord, s['q'], s['min_df'], s['max_df'])
            if lk not in self.cache:
                self.cache[lk] = kw.build_library(self.bank, self.base_cls, q=s['q'], coord=coord,
                                                  min_df=s['min_df'], max_df=s['max_df'])
            lib = self.cache[lk]
            if typ == 'keyword_random':
                rand_lib = lib.random_twin(seed=s['seed'])
        elif typ.startswith('freq') or typ in UNION_TYPES:
            deltas, common = self.freq_deltas(s['sigmas'])
        elif typ == 'prior':
            pass
        else:
            raise ValueError('unknown segment type ' + typ)
        fine = None
        if s['alpha_mod']:
            deltas_m, common_m = self.freq_deltas(s['sigmas'])
            fine = deltas_m[:, 0] - common_m[0]
        for i, j in enumerate(nov):
            c_id = int(self.all_cls[j])
            if lib is not None:
                words = kw.class_keywords(self.bank, c_id, lib, m=s['m'])
                use = rand_lib if rand_lib is not None else lib
                segs = [use.masks[w] for w in words]
            elif typ in UNION_TYPES:
                segs = [self._channel_prior(s, typ, deltas, common)]
            elif typ == 'prior':
                segs = [self._prior_mask(s, self.counts[j], tc, base_vis)]
            else:
                segs = self._freq_segments(j, c_id, s, deltas, common)
            if s['alpha_mod'] and lib is not None:
                alphas = frequency_alpha(s['alpha'], segs, fine[j], out[i], s['alpha_mod'],
                                         seed=c_id + 1)
            elif s['alpha_bands'] is not None and lib is None and typ not in UNION_TYPES + ('prior',):
                alphas = [s['alpha_bands'][b] for b in s['bands']]
            else:
                alphas = [s['alpha']] * len(segs)
            if s['sim'] == 'text':
                src_n, src_b = tc[j], tc[:self.nb]
            else:
                src_n, src_b = out[i], base_vis
            self._seg_cov.append(int(np.logical_or.reduce(segs).sum()) if segs else 0)
            self._seg_n.append(len(segs))
            out[i] = segment_step(out[i], segs, src_n, src_b, base_vis, s['r'], alphas, s['tau'],
                                  scope=s['scope'])
        return out

    def _channel_prior(self, s, typ, deltas, common):
        """One class-agnostic channel set shared by all novel classes (base-only).

        freq_union: union over the chosen bands of the top-q channels of the
        base-averaged |Delta_k - common_k|. The controls take the same number k
        of channels ranked by within-class variance (var_union), mean absolute
        activation (mag_union), Fisher ratio (fisher_union) or at random.
        """
        ck = _key('prior', typ, s['sigmas'], s['bands'], s['q'], s['seed'])
        if ck in self.cache:
            return self.cache[ck]
        U = np.logical_or.reduce([kw.topq_mask(np.abs(deltas[:self.nb, b] - common[b]).mean(0), s['q'])
                                  for b in s['bands']])
        k = int(U.sum())
        X, y = self.sup_feat[0], self.sup_lab[0]
        if typ == 'freq_union':
            m = U
        elif typ in ('var_union', 'fisher_union'):
            Xc, _ = within_class_centered(X, y, self.base_cls)
            wvar = (Xc ** 2).mean(0)
            if typ == 'var_union':
                m = kw.topq_mask(wvar, k)
            else:
                bvar = self.raw_mean[:self.nb].var(0)
                m = kw.topq_mask(bvar / np.maximum(wvar, 1e-12), k)
        elif typ == 'mag_union':
            m = kw.topq_mask(np.abs(X).mean(0), k)
        elif typ == 'random_union':
            m = np.zeros(self.bank.dim, dtype=bool)
            m[np.random.RandomState(12345 + int(s['seed'])).choice(self.bank.dim, k, replace=False)] = True
        else:
            raise ValueError(typ)
        self.cache[ck] = m
        return m

    def _freq_segments(self, j, c_id, s, deltas, common):
        q, D = int(s['q']), self.bank.dim
        typ = s['type']
        segs = []
        for b in s['bands']:
            if typ == 'freq':
                segs.append(kw.topq_mask(deltas[j, b] - common[b], q))
            elif typ == 'freq_classperm':
                donor = np.random.RandomState(c_id * 31 + b).randint(self.nb)
                segs.append(kw.topq_mask(deltas[donor, b] - common[b], q))
            elif typ == 'freq_common':
                ck = _key('freqcommon', s['sigmas'], b, q)
                if ck not in self.cache:
                    self.cache[ck] = kw.topq_mask(np.abs(deltas[:self.nb, b] - common[b]).mean(0), q)
                segs.append(self.cache[ck])
            elif typ == 'freq_random':
                r = np.random.RandomState(c_id * 131 + b + 7)
                m = np.zeros(D, dtype=bool)
                m[r.choice(D, min(q, D), replace=False)] = True
                segs.append(m)
        return segs

    # ------------------------------------------------------- BiMC reference
    def cos_full(self, protos, key):
        k = _key('cos', key)
        if k not in self.cache:
            self.cache[k] = self.Fq @ l2n(protos).T
        return self.cache[k]

    def bimc_cov_inv(self, t, gamma_base, gamma_inc):
        k = _key('covinv', t, gamma_base, gamma_inc)
        if k not in self.cache:
            covs, w = [], []
            for s in range(t + 1):
                X = self.sup_feat[s]
                cov = np.cov(X, rowvar=False) if len(X) > 1 else np.zeros((X.shape[1],) * 2)
                g = gamma_base if s == 0 else gamma_inc
                cov = cov + g * np.mean(np.diag(cov)) * np.eye(cov.shape[0])
                covs.append(cov)
                w.append(len(self.classes[s]))
            merged = sum(wi * ci for wi, ci in zip(w, covs)) / sum(w)
            self.cache[k] = np.linalg.pinv(merged)
        return self.cache[k]

    def maha_logits(self, pos, protos, P, key):
        k = _key('xpx', key)
        if k not in self.cache:
            self.cache[k] = ((self.Fq @ P) * self.Fq).sum(1)
        VP = protos @ P
        return -(self.cache[k][pos][:, None] - 2 * self.Fq[pos] @ VP.T + (VP * protos).sum(1)[None])

    def ref_probs(self, c, t, vis):
        cols = slice(0, self.n_cols[t])
        pos = self.qpos[t]
        if c['classifier'] == 'bimc':
            fused = c['beta'] * self.text_cal(c['lambda_t']) + (1 - c['beta']) * vis
            fk = _key('fused', c['beta'], c['lambda_t'], id(vis))
            cos = self.cos_full(fused, fk)
        elif c['classifier'] == 'proto':
            cos = self.cos_full(vis, _key('proto', id(vis)))
        elif c['classifier'] == 'zeroshot':
            cos = self.cos_full(self.text_cal(c['lambda_t']), _key('zs', c['lambda_t']))
        else:
            raise ValueError('unknown classifier ' + c['classifier'])
        p = softmax(cos[pos][:, cols])
        e = c['ensemble']
        if e is not None and e['alpha'] < 1:
            Pinv = self.bimc_cov_inv(t, e['gamma_base'], e['gamma_inc'])
            ml = self.maha_logits(pos, vis[:self.n_cols[t]], Pinv, (t, e['gamma_base'], e['gamma_inc']))
            p_cov = softmax(ml / 512.0)
            p_knn = softmax(self.knn_scores()[pos][:, cols])
            a = e['alpha']
            nb = self.nb
            p = np.concatenate([a * p[:, :nb] + (1 - a) * p_cov[:, :nb],
                                a * p[:, nb:] + (1 - a) * p_knn[:, nb:]], axis=1)
        return p

    # ---------------------------------------------------------------- anchor
    def _blocks(self, a):
        specs = expand_blocks(a['blocks'])
        k = _key('blocks', specs, a['eta'])
        if k not in self.cache:
            builders = [BlockBuilder(sp, eta=a['eta']).fit(self.bank, self.sup_rows[0],
                                                           self.sup_lab[0], self.base_cls)
                        for sp in specs]
            sup = [np.concatenate([self.sup_feat[t]] + [b.transform(self.bank, 'train', self.sup_rows[t])
                                                        for b in builders], axis=1)
                   for t in range(len(self.classes))]
            q = np.concatenate([self.Fq] + [b.transform(self.bank, self.ep.query_split, self.q_rows)
                                            for b in builders], axis=1)
            self.cache[k] = (builders, sup, q)
        return self.cache[k]

    def anchor_scores(self, a, vis):
        builders, sup, Zq = self._blocks(a)
        sizes = [self.bank.dim] + [b.dim for b in builders]
        gk = _key('gda', expand_blocks(a['blocks']), a['eta'], a['shrink'], a['ridge'], a['block_diag'])
        if gk not in self.cache:
            g = GDA(shrink=a['shrink'], ridge=a['ridge'], block_sizes=sizes,
                    block_diag=a['block_diag']).fit_base(sup[0], self.sup_lab[0], self.base_cls)
            self.cache[gk] = g
        g = self.cache[gk]
        means = np.zeros((len(self.all_cls), sup[0].shape[1]))
        for t, cls in enumerate(self.classes):
            for cc in cls:
                means[self.col[int(cc)]] = sup[t][self.sup_lab[t] == cc].mean(0)
        if a['mean_source'] == 'calibrated':
            nov = np.arange(self.nb, len(self.all_cls))
            d = self.bank.dim
            norms = np.linalg.norm(means[nov, :d], axis=1, keepdims=True)
            means[nov, :d] = l2n(vis[nov]) * norms
        if a['eb'] is not None and len(builders):
            means = self._eb_means(means, sup, a['eb'], sizes)
        sk = _key('ascore', gk, a['mean_source'], a['eb'], a['kappa'], a['kappa_mode'], id(vis))
        if sk not in self.cache:
            g.means, g.counts = [means], [self.counts]
            self.cache[sk] = g.scores(Zq, kappa=a['kappa'], mode=a['kappa_mode'])
        return self.cache[sk]

    def _eb_means(self, means, sup, eb, sizes):
        """R1: empirical-Bayes prior for novel block means from base-class offsets."""
        nb, d = self.nb, sizes[0]
        out = means.copy()
        Mo = means[:nb, :d]
        mo = Mo.mean(0)
        nov = np.arange(nb, len(self.all_cls))
        if not len(nov):
            return out
        Wn = l2n(Mo)
        start = d
        for k in sizes[1:]:
            sl = slice(start, start + k)
            start += k
            Mb = means[:nb, sl]
            mb = Mb.mean(0)
            H = ridge_fit(Mo - mo, Mb - mb, eb.get('eta', 1.0))
            E = Mb - (mb + (Mo - mo) @ H.T)
            xc, _ = within_class_centered(sup[0][:, sl], self.sup_lab[0], self.base_cls)
            s2w = float((xc ** 2).sum() / xc.size)
            sims = Wn @ Wn.T
            np.fill_diagonal(sims, -np.inf)
            topk = int(eb.get('topk', 5))
            errs = []
            for j in range(nb):
                top = np.argpartition(-sims[j], topk - 1)[:topk]
                w = softmax(eb.get('tau', 16.0) * sims[j, top])
                m = mb + (Mo[j] - mo) @ H.T + w @ E[top]
                errs.append(((Mb[j] - m) ** 2).mean())
            s2p = max(float(np.mean(errs)) - s2w / max(self.counts[:nb].mean(), 1), 1e-12)
            Vn = l2n(means[nov, :d])
            simn = Vn @ Wn.T
            for i, j in enumerate(nov):
                top = np.argpartition(-simn[i], topk - 1)[:topk]
                w = softmax(eb.get('tau', 16.0) * simn[i, top])
                m = mb + (means[j, :d] - mo) @ H.T + w @ E[top]
                rho = s2p / (s2p + s2w / self.counts[j])
                post = m + rho * (means[j, sl] - m)
                lam = eb.get('lam', 0.5)
                out[j, sl] = (1 - lam) * means[j, sl] + lam * post
        return out

    # ------------------------------------------------------------------ run
    def run(self, cfg, keep_scores=False):
        c = resolve_config(cfg, self.bank)
        vis = self.vis_cal(c)
        a = c['anchor']
        A = self.anchor_scores(a, vis) if a is not None and a['omega'] > 0 else None
        per_session, last = [], {}
        for t in range(len(self.classes)):
            p = self.ref_probs(c, t, vis)
            pos, cols = self.qpos[t], slice(0, self.n_cols[t])
            if A is not None:
                s = A[pos][:, cols] / float(a['temperature'])
                if a['fusion'] == 'prob':
                    z = np.log((1 - a['omega']) * p + a['omega'] * softmax(s) + 1e-300)
                elif a['fusion'] == 'logit':
                    z = np.log(p + 1e-300) + a['omega'] * log_softmax(s)
                elif a['fusion'] == 'gda_only':
                    z = s
                else:
                    raise ValueError('fusion must be prob, logit or gda_only')
            else:
                z = np.log(p + 1e-300)
            if c['novel_bias']:
                z = z.copy()
                z[:, self.nb:] += c['novel_bias']
            if c['ula'] is not None and t > 0:
                z = ula_adjust(z, self.nb, **c['ula'])
            pred = self.all_cls[:self.n_cols[t]][np.argmax(z, axis=1)]
            labels = self.q_lab[pos]
            per_session.append(session_metrics(pred, labels, self.base_cls))
            if t == len(self.classes) - 1:
                last = {'pred': pred, 'labels': labels}
                if keep_scores:
                    last['scores'] = z
        seg_stats = self.cache.get('segstats' + _key('vis', c['holistic'], c['segments'],
                                   c['lambda_t'] if self._uses_text(c) else None))
        return {'per_session': per_session, 'summary': summarize_sessions(per_session),
                'segment_stats': seg_stats,
                'last_pred': last['pred'], 'last_labels': last['labels'],
                'last_scores': last.get('scores'), 'config': c}


def run_episode(bank, episode, cfg, keep_scores=False):
    return EpisodeRunner(bank, episode).run(cfg, keep_scores=keep_scores)
