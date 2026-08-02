import csv
import json
import math
import os
from collections import defaultdict

import torch
import torch.nn.functional as F


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class FourierBandStop:
    """Create frequency counterfactuals by removing radial Fourier bands.

    Inputs are expected to have already been normalized with CLIP's RGB mean
    and standard deviation. The DC component is always retained so that a
    low-frequency intervention does not destroy the image's mean colour.
    """

    def __init__(self, band_names, band_edges):
        if len(band_edges) != len(band_names) + 1:
            raise ValueError("band_edges must contain exactly len(band_names) + 1 values")
        if band_edges[0] != 0.0 or band_edges[-1] != 1.0:
            raise ValueError("band_edges must start at 0.0 and end at 1.0")
        if any(left >= right for left, right in zip(band_edges[:-1], band_edges[1:])):
            raise ValueError("band_edges must be strictly increasing")

        self.band_names = list(band_names)
        self.band_edges = [float(edge) for edge in band_edges]
        self._mask_cache = {}

    def _masks(self, height, width, device):
        key = (height, width, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]

        fy = torch.fft.fftfreq(height, device=device)
        fx = torch.fft.fftfreq(width, device=device)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        max_radius = math.sqrt(0.5 ** 2 + 0.5 ** 2)
        radius = torch.sqrt(xx.square() + yy.square()) / max_radius

        masks = []
        for index, (lower, upper) in enumerate(zip(self.band_edges[:-1], self.band_edges[1:])):
            if index == len(self.band_names) - 1:
                mask = (radius >= lower) & (radius <= upper)
            else:
                mask = (radius >= lower) & (radius < upper)
            mask[0, 0] = False
            masks.append(mask.unsqueeze(0).unsqueeze(0))

        self._mask_cache[key] = masks
        return masks

    def remove(self, normalized_images, band_index):
        if normalized_images.ndim != 4 or normalized_images.shape[1] != 3:
            raise ValueError("normalized_images must have shape [N, 3, H, W]")
        if not 0 <= band_index < len(self.band_names):
            raise IndexError("band_index is out of range")

        input_dtype = normalized_images.dtype
        work_images = normalized_images.float()
        mean = work_images.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std = work_images.new_tensor(CLIP_STD).view(1, 3, 1, 1)
        rgb_images = (work_images * std + mean).clamp(0.0, 1.0)

        spectrum = torch.fft.fft2(rgb_images, dim=(-2, -1), norm="ortho")
        mask = self._masks(rgb_images.shape[-2], rgb_images.shape[-1], rgb_images.device)[band_index]
        counterfactual = torch.fft.ifft2(
            spectrum.masked_fill(mask, 0.0), dim=(-2, -1), norm="ortho"
        ).real.clamp(0.0, 1.0)
        counterfactual = (counterfactual - mean) / std
        return counterfactual.to(dtype=input_dtype)


def semantic_margins(image_features, text_features, labels):
    """Return correct-class cosine similarity minus the strongest competitor."""
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)
    labels = labels.long()

    if text_features.shape[0] < 2:
        raise ValueError("semantic margin requires at least two candidate classes")
    if labels.numel() and int(labels.max()) >= text_features.shape[0]:
        raise ValueError("labels must index rows in text_features")

    similarities = image_features @ text_features.T
    correct = similarities.gather(1, labels.unsqueeze(1)).squeeze(1)
    competitors = similarities.clone()
    competitors.scatter_(1, labels.unsqueeze(1), float("-inf"))
    strongest_competitor = competitors.max(dim=1).values
    return correct - strongest_competitor


class FrequencyContributionAccumulator:
    def __init__(self, band_names, weight_temperature=0.05):
        if weight_temperature <= 0:
            raise ValueError("weight_temperature must be positive")
        self.band_names = list(band_names)
        self.weight_temperature = float(weight_temperature)
        self._values = defaultdict(lambda: defaultdict(list))

    def update(self, labels, contributions):
        labels = labels.detach().cpu().long()
        for band_name in self.band_names:
            if band_name not in contributions:
                raise KeyError("missing contribution for band '{}'".format(band_name))
            values = contributions[band_name].detach().cpu().float()
            if values.shape[0] != labels.shape[0]:
                raise ValueError("labels and contributions must have the same batch size")
            for label, value in zip(labels.tolist(), values.tolist()):
                self._values[int(label)][band_name].append(float(value))

    def records(self, class_names, task_id):
        records = []
        for class_id in sorted(self._values):
            band_statistics = {}
            means = []
            sample_count = None
            for band_name in self.band_names:
                values = torch.tensor(self._values[class_id][band_name], dtype=torch.float32)
                if values.numel() == 0:
                    raise RuntimeError("class {} has no samples for band {}".format(class_id, band_name))
                sample_count = int(values.numel())
                std = values.std(unbiased=False)
                mean = values.mean()
                means.append(mean)
                band_statistics[band_name] = {
                    "mean": float(mean),
                    "std": float(std),
                    "sem": float(std / math.sqrt(sample_count)),
                    "positive_fraction": float((values > 0).float().mean()),
                }

            mean_tensor = torch.stack(means)
            weights = torch.softmax(mean_tensor / self.weight_temperature, dim=0)
            dominant_index = int(torch.argmax(mean_tensor))
            record = {
                "task_id": int(task_id),
                "session_type": "base" if task_id == 0 else "incremental",
                "class_id": int(class_id),
                "class_name": str(class_names[class_id]),
                "num_samples": sample_count,
                "dominant_band": self.band_names[dominant_index],
                "contribution_spread": float(mean_tensor.max() - mean_tensor.min()),
                "preference_strength": float(weights.max() - weights.min()),
                "bands": band_statistics,
                "frequency_weights": {
                    name: float(weight) for name, weight in zip(self.band_names, weights)
                },
            }
            records.append(record)
        return records


def summarize_frequency_records(records, band_names):
    if not records:
        return {
            "num_classes": 0,
            "dominant_band_counts": {name: 0 for name in band_names},
        }

    dominant_counts = {name: 0 for name in band_names}
    contribution_profiles = []
    weight_profiles = []
    for record in records:
        dominant_counts[record["dominant_band"]] += 1
        contribution_profiles.append([record["bands"][name]["mean"] for name in band_names])
        weight_profiles.append([record["frequency_weights"][name] for name in band_names])

    contributions = torch.tensor(contribution_profiles, dtype=torch.float32)
    weights = torch.tensor(weight_profiles, dtype=torch.float32)
    if len(records) > 1:
        mean_pairwise_weight_l1 = float(torch.pdist(weights, p=1).mean())
    else:
        mean_pairwise_weight_l1 = 0.0

    class_effect_eta_squared = {}
    for band_index, band_name in enumerate(band_names):
        sample_counts = torch.tensor(
            [record["num_samples"] for record in records], dtype=torch.float32
        )
        class_means = contributions[:, band_index]
        grand_mean = (sample_counts * class_means).sum() / sample_counts.sum()
        ss_between = (sample_counts * (class_means - grand_mean).square()).sum()
        ss_within = sum(
            record["num_samples"] * record["bands"][band_name]["std"] ** 2
            for record in records
        )
        ss_total = float(ss_between) + float(ss_within)
        class_effect_eta_squared[band_name] = (
            float(ss_between) / ss_total if ss_total > 0 else 0.0
        )

    return {
        "num_classes": len(records),
        "dominant_band_counts": dominant_counts,
        "band_mean_contribution_across_classes": {
            name: float(contributions[:, index].mean()) for index, name in enumerate(band_names)
        },
        "band_std_contribution_across_classes": {
            name: float(contributions[:, index].std(unbiased=False)) for index, name in enumerate(band_names)
        },
        "band_mean_weight_across_classes": {
            name: float(weights[:, index].mean()) for index, name in enumerate(band_names)
        },
        "mean_preference_strength": float(
            sum(record["preference_strength"] for record in records) / len(records)
        ),
        "mean_pairwise_frequency_weight_l1": mean_pairwise_weight_l1,
        "class_effect_eta_squared": class_effect_eta_squared,
    }


def write_frequency_records(output_dir, stem, records, summary, metadata):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, stem + ".json")
    csv_path = os.path.join(output_dir, stem + ".csv")

    payload = {
        "metadata": metadata,
        "summary": summary,
        "classes": records,
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    band_names = list(metadata["band_names"])
    fieldnames = [
        "task_id",
        "session_type",
        "class_id",
        "class_name",
        "num_samples",
        "dominant_band",
        "contribution_spread",
        "preference_strength",
    ]
    for band_name in band_names:
        fieldnames.extend([
            band_name + "_mean",
            band_name + "_std",
            band_name + "_sem",
            band_name + "_positive_fraction",
            band_name + "_weight",
        ])

    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fieldnames if key in record}
            for band_name in band_names:
                stats = record["bands"][band_name]
                row[band_name + "_mean"] = stats["mean"]
                row[band_name + "_std"] = stats["std"]
                row[band_name + "_sem"] = stats["sem"]
                row[band_name + "_positive_fraction"] = stats["positive_fraction"]
                row[band_name + "_weight"] = record["frequency_weights"][band_name]
            writer.writerow(row)

    return json_path, csv_path
