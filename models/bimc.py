import torch
import torch.nn as nn
import torch.nn.functional as F
import models.clip.clip as clip
import json

from models.frequency import (
    RadialFrequencyDecomposer,
    calibrate_frequency_prototypes,
    compute_frequency_band_logits,
    compute_frequency_class_alpha,
    compute_frequency_logits,
    compute_frequency_prototypes,
    mix_frequency_probabilities,
    route_description_embeddings,
    select_frequency_description_prototypes,
)
from models.frequency_router import (
    TrainableFrequencyRouter,
    residual_frequency_fusion,
    sample_pseudo_fscil_episode,
)
from models.frequency_modality import (
    FrequencyModalityEncoder,
    FrequencySpectrumDescriptor,
    compute_modality_alpha,
    compute_modality_prototypes,
    fuse_modality_probabilities,
)

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class BiMC(nn.Module):

    def __init__(self, cfg, template, device):
        super(BiMC, self).__init__()
        self.cfg = cfg
        self.device = device
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        print(f"Prompt template:{template}")
        self.template = template
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.BiMC.PREC == "fp32" or cfg.TRAINER.BiMC.PREC == "amp":
        # CLIP's default precision is fp16
            clip_model.float()

        clip_model.eval()
        self.clip_model = clip_model.to(self.device)
        self.text_proto = None
        self.description_proto = None
        self.vision_proto = None
        self.frequency_enabled = cfg.TRAINER.BiMC.FREQUENCY.ENABLED
        self.frequency_decomposer = None
        self.frequency_router = None
        if self.frequency_enabled:
            frequency_cfg = cfg.TRAINER.BiMC.FREQUENCY
            if len(frequency_cfg.PROMPTS) != 3:
                raise ValueError("BiMC frequency mode currently expects three prompts/bands.")
            if len(frequency_cfg.BAND_PRIOR) != 3:
                raise ValueError("BiMC frequency mode expects three BAND_PRIOR values.")
            if not 0.0 <= frequency_cfg.FREQ_ALPHA <= 1.0:
                raise ValueError("FREQ_ALPHA must lie in [0, 1].")
            if not 0.0 <= frequency_cfg.DESCRIPTION_WEIGHT <= 1.0:
                raise ValueError("DESCRIPTION_WEIGHT must lie in [0, 1].")
            if frequency_cfg.DESCRIPTION_TOPK <= 0:
                raise ValueError("DESCRIPTION_TOPK must be positive.")
            if frequency_cfg.DESCRIPTION_TEMPERATURE <= 0.0:
                raise ValueError("DESCRIPTION_TEMPERATURE must be positive.")
            if (
                frequency_cfg.USE_EXPLICIT_DESCRIPTIONS
                and not frequency_cfg.EXPLICIT_DESCRIPTION_PATH
            ):
                raise ValueError(
                    "EXPLICIT_DESCRIPTION_PATH is required when explicit "
                    "frequency descriptions are enabled."
                )
            self.frequency_decomposer = RadialFrequencyDecomposer(
                low_cutoff=frequency_cfg.LOW_CUTOFF,
                high_cutoff=frequency_cfg.HIGH_CUTOFF,
                center_residual_bands=frequency_cfg.CENTER_RESIDUAL_BANDS,
                fft_batch_size=frequency_cfg.FFT_BATCH_SIZE,
                fft_device=frequency_cfg.FFT_DEVICE,
                view_mode=frequency_cfg.VIEW_MODE,
                high_enhance=frequency_cfg.HIGH_ENHANCE,
            ).to(self.device)
            router_cfg = frequency_cfg.ROUTER
            if router_cfg.ENABLED:
                if router_cfg.TRAIN_STEPS <= 0:
                    raise ValueError("Router TRAIN_STEPS must be positive.")
                if router_cfg.EPISODE_WAY < 2:
                    raise ValueError("Router EPISODE_WAY must be at least two.")
                if not 0 <= router_cfg.EPISODE_OLD_WAY <= router_cfg.EPISODE_WAY:
                    raise ValueError(
                        "Router EPISODE_OLD_WAY must lie in [0, EPISODE_WAY]."
                    )
                if router_cfg.OLD_SHOT <= 0:
                    raise ValueError("Router OLD_SHOT must be positive.")
                if router_cfg.SHOT <= 0 or router_cfg.QUERY <= 0:
                    raise ValueError("Router SHOT and QUERY must be positive.")
                if router_cfg.LR <= 0.0:
                    raise ValueError("Router learning rate must be positive.")
                if router_cfg.ORACLE_TEMPERATURE <= 0.0:
                    raise ValueError(
                        "Router ORACLE_TEMPERATURE must be positive."
                    )
                if router_cfg.LOG_INTERVAL <= 0:
                    raise ValueError("Router LOG_INTERVAL must be positive.")
                if (
                    router_cfg.ROUTE_LOSS_WEIGHT < 0.0
                    or router_cfg.SAFE_KL_WEIGHT < 0.0
                    or router_cfg.NOVEL_LOSS_WEIGHT < 0.0
                ):
                    raise ValueError("Router loss weights must be non-negative.")
                if not 0.0 <= router_cfg.MAX_ALPHA <= 1.0:
                    raise ValueError("Router MAX_ALPHA must lie in [0, 1].")
                self.frequency_router = TrainableFrequencyRouter(
                    num_bands=self.frequency_decomposer.num_bands,
                    hidden_dim=router_cfg.HIDDEN_DIM,
                    dropout=router_cfg.DROPOUT,
                    null_logit_bias=router_cfg.NULL_LOGIT_BIAS,
                ).to(self.device)

        self.frequency_modality_enabled = (
            cfg.TRAINER.BiMC.FREQUENCY_MODALITY.ENABLED
        )
        self.frequency_modality_descriptor = None
        self.frequency_modality_encoder = None
        self.frequency_modality_fitted = False
        if self.frequency_modality_enabled:
            modality_cfg = cfg.TRAINER.BiMC.FREQUENCY_MODALITY
            if modality_cfg.TRAIN_STEPS <= 0 or modality_cfg.BATCH_SIZE <= 0:
                raise ValueError(
                    "Frequency modality TRAIN_STEPS and BATCH_SIZE must be positive."
                )
            if modality_cfg.LR <= 0.0 or modality_cfg.LOGIT_SCALE <= 0.0:
                raise ValueError(
                    "Frequency modality LR and LOGIT_SCALE must be positive."
                )
            if modality_cfg.LOG_INTERVAL <= 0 or modality_cfg.TEMPERATURE <= 0.0:
                raise ValueError(
                    "Frequency modality LOG_INTERVAL and TEMPERATURE must be positive."
                )
            if not 0.0 <= modality_cfg.MIN_FUSION_WEIGHT <= (
                modality_cfg.FUSION_WEIGHT
            ) <= 1.0:
                raise ValueError(
                    "Frequency modality weights must satisfy 0 <= min <= fusion <= 1."
                )
            if (
                modality_cfg.CLASSIFICATION_WEIGHT < 0.0
                or modality_cfg.IMAGE_ALIGNMENT_WEIGHT < 0.0
                or modality_cfg.CLASSIFICATION_WEIGHT
                + modality_cfg.IMAGE_ALIGNMENT_WEIGHT
                <= 0.0
            ):
                raise ValueError(
                    "At least one frequency modality training loss must be positive."
                )
            self.frequency_modality_descriptor = FrequencySpectrumDescriptor(
                grid_size=modality_cfg.GRID_SIZE,
                radial_bins=modality_cfg.RADIAL_BINS,
                fft_batch_size=modality_cfg.FFT_BATCH_SIZE,
                fft_device=modality_cfg.FFT_DEVICE,
            ).to(self.device)
            clip_dim = int(self.clip_model.text_projection.shape[1])
            self.frequency_modality_encoder = FrequencyModalityEncoder(
                input_dim=self.frequency_modality_descriptor.descriptor_dim,
                output_dim=clip_dim,
                hidden_dim=modality_cfg.HIDDEN_DIM,
                dropout=modality_cfg.DROPOUT,
            ).to(self.device)


    @torch.no_grad()
    def inference_text_feature(self, class_names, template, cls_begin_index):
        print(f'class names: {class_names}')
        clip_weights = []
        all_targets = []
        k = cls_begin_index
        for classname in class_names:
            targets = torch.full((len(template),), k)
            all_targets.append(targets)
            k += 1
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            classname = classname.replace('-', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(self.device)
            # prompt ensemble for ImageNet
            class_embeddings = self.clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)
        clip_weights = torch.stack(clip_weights, dim=0)
        clip_weights = F.normalize(clip_weights, dim=-1)
        all_targets = torch.cat(all_targets, dim=0)
        return clip_weights, all_targets


    @torch.no_grad()
    def inference_frequency_text_feature(self, class_names):
        """Encode one structured semantic prompt for each frequency band."""
        prompt_templates = self.cfg.TRAINER.BiMC.FREQUENCY.PROMPTS
        class_embeddings = []
        for classname in class_names:
            classname = classname.replace('_', ' ').replace('-', ' ')
            texts = [template.format(classname) for template in prompt_templates]
            tokens = clip.tokenize(texts).to(self.device)
            embeddings = self.clip_model.encode_text(tokens)
            class_embeddings.append(F.normalize(embeddings, dim=-1))
        return torch.stack(class_embeddings, dim=0)


    @staticmethod
    def _normalize_description_key(name):
        return " ".join(
            str(name).lower().replace('_', ' ').replace('-', ' ').split()
        )


    @torch.no_grad()
    def inference_explicit_frequency_description_candidates(
        self, class_names, description_path
    ):
        """Encode explicit CUB-style descriptions as ``[C, B, K, D]``."""
        try:
            with open(description_path, "r", encoding="utf-8") as file:
                prompt_dict = json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                "Explicit frequency description file was not found: {}. "
                "Run tools/generate_cub200_frequency_descriptions.py first."
                .format(description_path)
            ) from error
        if not isinstance(prompt_dict, dict) or not prompt_dict:
            raise ValueError(
                "Explicit frequency descriptions must be a non-empty JSON object."
            )

        normalized_prompts = {}
        for key, value in prompt_dict.items():
            normalized_key = self._normalize_description_key(key)
            if normalized_key in normalized_prompts:
                raise ValueError(
                    "Duplicate normalized class key in explicit descriptions: {}"
                    .format(key)
                )
            normalized_prompts[normalized_key] = value

        band_names = ("low", "middle", "high")
        class_embeddings = []
        expected_candidates = None
        for classname in class_names:
            normalized_key = self._normalize_description_key(classname)
            if normalized_key not in normalized_prompts:
                raise KeyError(
                    "Missing explicit frequency descriptions for class: {}"
                    .format(classname)
                )
            class_prompts = normalized_prompts[normalized_key]
            if not isinstance(class_prompts, dict) or set(class_prompts) != set(
                band_names
            ):
                raise ValueError(
                    "Class {} must contain exactly low, middle, and high lists."
                    .format(classname)
                )

            band_embeddings = []
            for band_name in band_names:
                descriptions = class_prompts[band_name]
                if (
                    not isinstance(descriptions, list)
                    or not descriptions
                    or not all(
                        isinstance(description, str) and description.strip()
                        for description in descriptions
                    )
                ):
                    raise ValueError(
                        "Descriptions for {}/{} must be a non-empty string list."
                        .format(classname, band_name)
                    )
                if expected_candidates is None:
                    expected_candidates = len(descriptions)
                elif len(descriptions) != expected_candidates:
                    raise ValueError(
                        "All class/band entries must have the same number of "
                        "candidates; {}/{} has {}, expected {}."
                        .format(
                            classname,
                            band_name,
                            len(descriptions),
                            expected_candidates,
                        )
                    )
                tokens = clip.tokenize(descriptions).to(self.device)
                embeddings = self.clip_model.encode_text(tokens)
                band_embeddings.append(F.normalize(embeddings, dim=-1))
            class_embeddings.append(torch.stack(band_embeddings, dim=0))
        return torch.stack(class_embeddings, dim=0)


    @torch.no_grad()
    def inference_all_img_feature(self, loader, cls_begin_index, class_index=None):
        all_features = []
        all_labels = []
        all_frequency_features = []
        all_frequency_modality_descriptors = []
        for batch in loader:
            images, labels = self.parse_batch(batch)
            features = self.clip_model.encode_image(images)
            features = F.normalize(features, dim=-1)
            all_features.append(features)
            all_labels.append(labels)
            if self.frequency_enabled:
                all_frequency_features.append(
                    self.extract_frequency_img_feature(images)
                )
            if self.frequency_modality_enabled:
                all_frequency_modality_descriptors.append(
                    self.extract_frequency_modality_descriptor(images)
                )
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        if class_index is None:
            ordered_labels = [int(label) for label in torch.unique(all_labels)]
        else:
            ordered_labels = [int(label) for label in class_index]
        print(f'all targets:{torch.as_tensor(ordered_labels)}')
        prototypes = []
        for class_id in ordered_labels:
            idx = torch.where(class_id == all_labels)[0]
            if idx.numel() == 0:
                raise ValueError("No image features found for class {}.".format(class_id))
            class_features = all_features[idx]
            class_prototype = class_features.mean(dim=0)
            prototypes.append(class_prototype)
        prototypes = torch.stack(prototypes, dim=0)
        prototypes = F.normalize(prototypes, dim=-1)

        if self.frequency_modality_enabled:
            frequency_modality_descriptors = torch.cat(
                all_frequency_modality_descriptors, dim=0
            )
        else:
            frequency_modality_descriptors = None

        if not self.frequency_enabled:
            return (
                all_features,
                all_labels,
                prototypes,
                None,
                None,
                None,
                None,
                frequency_modality_descriptors,
            )

        all_frequency_features = torch.cat(all_frequency_features, dim=0)
        (
            frequency_prototypes,
            frequency_uncertainty,
            frequency_sample_counts,
        ) = compute_frequency_prototypes(
            all_frequency_features,
            all_labels,
            ordered_labels,
            return_counts=True,
        )
        return (
            all_features,
            all_labels,
            prototypes,
            frequency_prototypes,
            frequency_uncertainty,
            frequency_sample_counts,
            all_frequency_features,
            frequency_modality_descriptors,
        )


    @torch.no_grad()
    def inference_all_description_feature(self, class_names, gpt_path, cls_begin_index):
        description_embeddings = []
        mean_embeddings = []
        frequency_description_embeddings = []
        all_targets = []
        with open(gpt_path, "r") as file:
            GPT_prompt_dict = json.load(file)
        # The order of embeddings should follow strictly order of classname variable
        # Keys name should match classnames so that we could do fetching from the dict.
        # Convert the dict to lower case
        GPT_prompt_dict = {k.lower().replace("_", " "): v for k, v in GPT_prompt_dict.items()}
        k = cls_begin_index
        for single_key in class_names:
            single_class_prompts = GPT_prompt_dict[single_key.lower().replace("_", " ")]
            targets = torch.full((len(single_class_prompts),), k)

            k += 1
            x_tokenized = torch.cat([clip.tokenize(p) for p in single_class_prompts])
            with torch.no_grad():
                text_features = self.clip_model.encode_text(x_tokenized.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_embeddings.append(text_features.mean(0).unsqueeze(0))
            description_embeddings.append(text_features)
            all_targets.append(targets)
            if (
                self.frequency_enabled
                and not self.cfg.TRAINER.BiMC.FREQUENCY.USE_EXPLICIT_DESCRIPTIONS
            ):
                frequency_cfg = self.cfg.TRAINER.BiMC.FREQUENCY
                keyword_groups = (
                    frequency_cfg.LOW_KEYWORDS,
                    frequency_cfg.MIDDLE_KEYWORDS,
                    frequency_cfg.HIGH_KEYWORDS,
                )
                frequency_description_embeddings.append(
                    route_description_embeddings(
                        single_class_prompts, text_features, keyword_groups
                    )
                )
        description_embeddings = torch.cat(description_embeddings, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mean_embeddings = torch.cat(mean_embeddings, dim=0)
        mean_embeddings = F.normalize(mean_embeddings, dim=-1)
        if (
            self.frequency_enabled
            and not self.cfg.TRAINER.BiMC.FREQUENCY.USE_EXPLICIT_DESCRIPTIONS
        ):
            frequency_description_embeddings = torch.stack(
                frequency_description_embeddings, dim=0
            )
        else:
            frequency_description_embeddings = None
        return (
            description_embeddings,
            all_targets,
            mean_embeddings,
            frequency_description_embeddings,
        )


    def soft_calibration(self, base_protos, cur_protos):
        shift_weight = self.cfg.TRAINER.BiMC.LAMBDA_I
        tau = self.cfg.TRAINER.BiMC.TAU
        base_protos = F.normalize(base_protos, p=2, dim=-1)
        cur_protos = F.normalize(cur_protos, p=2, dim=-1)
        weights = torch.mm(cur_protos, base_protos.T) * tau
        norm_weights = torch.softmax(weights, dim=1)
        delta_protos = torch.matmul(norm_weights, base_protos)
        delta_protos = F.normalize(delta_protos, p=2, dim=-1)
        updated_protos = (1 - shift_weight) * cur_protos + shift_weight * delta_protos
        updated_protos = F.normalize(updated_protos, dim=-1)
        return updated_protos


    def fit_frequency_modality_encoder(
        self,
        descriptors,
        image_features,
        labels,
        class_index,
        text_features,
    ):
        """Train the independent spectral projection on base-session data.

        CLIP remains frozen.  The classification loss aligns spectra with the
        base text prototypes, while the instance loss transfers complementary
        semantic structure from CLIP image embeddings.  The encoder is frozen
        immediately after this one-time fit.
        """
        if not self.frequency_modality_enabled:
            return
        if self.frequency_modality_fitted:
            raise RuntimeError("Frequency modality encoder was already fitted.")
        modality_cfg = self.cfg.TRAINER.BiMC.FREQUENCY_MODALITY
        descriptors = descriptors.detach().float()
        image_features = F.normalize(image_features.detach().float(), dim=-1)
        text_features = F.normalize(text_features.detach().float(), dim=-1)
        labels = labels.detach().long()
        class_ids = [int(class_id) for class_id in class_index]

        local_labels = torch.full_like(labels, -1)
        for local_id, global_id in enumerate(class_ids):
            local_labels[labels == global_id] = local_id
        if torch.any(local_labels < 0):
            raise ValueError("Base labels do not match frequency modality classes.")
        if text_features.shape[0] != len(class_ids):
            raise ValueError("Text prototypes do not match frequency modality classes.")

        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.frequency_modality_encoder.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.AdamW(
            self.frequency_modality_encoder.parameters(),
            lr=modality_cfg.LR,
            weight_decay=modality_cfg.WEIGHT_DECAY,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.cfg.SEED) + 2903)
        self.frequency_modality_encoder.train()

        num_samples = descriptors.shape[0]
        batch_size = min(int(modality_cfg.BATCH_SIZE), num_samples)
        print(
            "train independent frequency modality: steps={}, samples={}, "
            "descriptor_dim={}".format(
                modality_cfg.TRAIN_STEPS,
                num_samples,
                descriptors.shape[1],
            )
        )
        running_loss = 0.0
        running_correct = 0
        running_count = 0
        running_steps = 0
        for step in range(1, modality_cfg.TRAIN_STEPS + 1):
            indices = torch.randint(
                num_samples,
                (batch_size,),
                generator=generator,
                device="cpu",
            ).to(descriptors.device)
            frequency_features = self.frequency_modality_encoder(
                descriptors[indices]
            )
            logits = (
                modality_cfg.LOGIT_SCALE
                * frequency_features
                @ text_features.t()
            )
            classification_loss = F.cross_entropy(
                logits, local_labels[indices]
            )
            alignment_loss = (
                1.0
                - F.cosine_similarity(
                    frequency_features, image_features[indices], dim=-1
                )
            ).mean()
            loss = (
                modality_cfg.CLASSIFICATION_WEIGHT * classification_loss
                + modality_cfg.IMAGE_ALIGNMENT_WEIGHT * alignment_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach())
            running_correct += int(
                (logits.argmax(dim=-1) == local_labels[indices]).sum()
            )
            running_count += batch_size
            running_steps += 1
            if (
                step == 1
                or step == modality_cfg.TRAIN_STEPS
                or step % modality_cfg.LOG_INTERVAL == 0
            ):
                print(
                    "frequency modality step {}/{}: loss={:.4f}, "
                    "text-prototype acc={:.2f}%".format(
                        step,
                        modality_cfg.TRAIN_STEPS,
                        running_loss / running_steps,
                        100.0 * running_correct / running_count,
                    )
                )
                running_loss = 0.0
                running_correct = 0
                running_count = 0
                running_steps = 0

        self.frequency_modality_encoder.eval()
        for parameter in self.frequency_modality_encoder.parameters():
            parameter.requires_grad_(False)
        self.frequency_modality_fitted = True
        print("independent frequency modality training complete; encoder frozen")


    def fit_frequency_router(
        self,
        image_features,
        frequency_features,
        labels,
        class_index,
        text_features,
        description_proto,
        frequency_semantic_proto,
    ):
        """Meta-train the residual router on pseudo FSCIL base episodes.

        The CLIP encoders and all prototypes are treated as frozen feature
        generators.  Only the small router MLP receives gradients.  Each
        episode contains pseudo-old classes with more support samples and
        pseudo-novel classes with the configured few-shot support size.
        """
        if self.frequency_router is None:
            return
        if frequency_features is None:
            raise ValueError("Router training requires per-image frequency features.")

        router_cfg = self.cfg.TRAINER.BiMC.FREQUENCY.ROUTER
        frequency_cfg = self.cfg.TRAINER.BiMC.FREQUENCY
        class_ids = [int(class_id) for class_id in class_index]
        if len(class_ids) < 2:
            raise ValueError("Router training requires at least two base classes.")

        # CLIP is frozen.  Detaching once prevents accidental graph retention
        # and keeps the pseudo-episode loop inexpensive.
        image_features = F.normalize(image_features.detach(), dim=-1)
        frequency_features = F.normalize(
            frequency_features.detach(), dim=-1
        )
        labels = labels.detach()
        text_features = F.normalize(text_features.detach(), dim=-1)
        description_proto = F.normalize(description_proto.detach(), dim=-1)
        frequency_semantic_proto = F.normalize(
            frequency_semantic_proto.detach(), dim=-1
        )

        optimizer = torch.optim.AdamW(
            self.frequency_router.parameters(),
            lr=router_cfg.LR,
            weight_decay=router_cfg.WEIGHT_DECAY,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.cfg.SEED) + 1701)
        self.frequency_router.train()

        lambda_t = (
            self.cfg.TRAINER.BiMC.LAMBDA_T
            if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION
            else 0.0
        )
        beta = self.cfg.DATASET.BETA
        running_loss = 0.0
        running_original_correct = 0
        running_mixed_correct = 0
        running_count = 0
        running_router = None
        running_steps = 0

        print(
            "train frequency residual router: steps={}, episode={} way "
            "({} pseudo-old), old-shot={}, novel-shot={}, query={}".format(
                router_cfg.TRAIN_STEPS,
                router_cfg.EPISODE_WAY,
                router_cfg.EPISODE_OLD_WAY,
                router_cfg.OLD_SHOT,
                router_cfg.SHOT,
                router_cfg.QUERY,
            )
        )

        for step in range(1, router_cfg.TRAIN_STEPS + 1):
            (
                selected_positions_cpu,
                support_indices_cpu,
                support_local_labels_cpu,
                query_indices_cpu,
                query_local_labels_cpu,
            ) = sample_pseudo_fscil_episode(
                labels=labels,
                class_ids=class_ids,
                way=router_cfg.EPISODE_WAY,
                shot=router_cfg.SHOT,
                query=router_cfg.QUERY,
                generator=generator,
                old_way=router_cfg.EPISODE_OLD_WAY,
                old_shot=router_cfg.OLD_SHOT,
            )

            device = image_features.device
            selected_positions = selected_positions_cpu.to(device)
            support_indices = support_indices_cpu.to(device)
            support_local_labels = support_local_labels_cpu.to(device)
            query_indices = query_indices_cpu.to(device)
            query_local_labels = query_local_labels_cpu.to(device)
            episode_way = selected_positions.numel()

            with torch.no_grad():
                support_image = image_features[support_indices]
                query_image = image_features[query_indices]
                support_frequency = frequency_features[support_indices]
                query_frequency = frequency_features[query_indices]

                image_prototypes = []
                for local_class in range(episode_way):
                    class_support = support_image[
                        support_local_labels == local_class
                    ]
                    image_prototypes.append(
                        F.normalize(class_support.mean(dim=0), dim=-1)
                    )
                image_prototypes = torch.stack(image_prototypes, dim=0)

                (
                    frequency_prototypes,
                    frequency_uncertainty,
                ) = compute_frequency_prototypes(
                    support_frequency,
                    support_local_labels,
                    class_index=range(episode_way),
                )
                semantic_prototypes = frequency_semantic_proto[
                    selected_positions
                ]
                (
                    calibrated_frequency_prototypes,
                    episode_band_weights,
                    _,
                    _,
                ) = calibrate_frequency_prototypes(
                    visual_prototypes=frequency_prototypes,
                    semantic_prototypes=semantic_prototypes,
                    uncertainty=frequency_uncertainty,
                    semantic_weight=frequency_cfg.SEMANTIC_WEIGHT,
                    max_semantic_weight=frequency_cfg.MAX_SEMANTIC_WEIGHT,
                    uncertainty_scale=frequency_cfg.UNCERTAINTY_SCALE,
                    alignment_scale=frequency_cfg.ALIGNMENT_SCALE,
                    fusion_temperature=frequency_cfg.FUSION_TEMPERATURE,
                    adaptive_fusion=frequency_cfg.ADAPTIVE_FUSION,
                    semantic_gate_mode=frequency_cfg.SEMANTIC_GATE_MODE,
                    band_prior=torch.as_tensor(
                        frequency_cfg.BAND_PRIOR,
                        device=device,
                        dtype=frequency_prototypes.dtype,
                    ),
                )

                episode_text = text_features[selected_positions]
                episode_description = description_proto[selected_positions]
                fused_prototypes = beta * (
                    (1.0 - lambda_t) * episode_text
                    + lambda_t * episode_description
                ) + (1.0 - beta) * image_prototypes
                fused_prototypes = F.normalize(fused_prototypes, dim=-1)
                original_logits = query_image @ fused_prototypes.t()
                band_logits = compute_frequency_band_logits(
                    query_frequency,
                    calibrated_frequency_prototypes,
                )

                # The best expert is known only for base-session query labels.
                # It supplies a soft routing target that can generalize to
                # unlabeled real incremental queries.
                expert_logits = torch.cat(
                    (original_logits.unsqueeze(-1), band_logits), dim=-1
                )
                expert_log_probabilities = F.log_softmax(
                    expert_logits.float(), dim=1
                )
                gather_labels = query_local_labels.view(-1, 1, 1).expand(
                    -1, 1, expert_logits.shape[-1]
                )
                expert_losses = -torch.gather(
                    expert_log_probabilities,
                    dim=1,
                    index=gather_labels,
                ).squeeze(1)
                oracle_router = F.softmax(
                    -expert_losses / router_cfg.ORACLE_TEMPERATURE,
                    dim=-1,
                )

            router_probabilities = self.frequency_router(
                original_logits, band_logits
            )
            mixed_logits = residual_frequency_fusion(
                original_logits=original_logits,
                band_logits=band_logits,
                router_probabilities=router_probabilities,
                class_band_weights=episode_band_weights,
                max_alpha=router_cfg.MAX_ALPHA,
            )

            classification_loss = F.cross_entropy(
                mixed_logits, query_local_labels
            )
            novel_start = min(
                int(router_cfg.EPISODE_OLD_WAY), episode_way
            )
            novel_mask = query_local_labels >= novel_start
            if novel_start < episode_way and torch.any(novel_mask):
                novel_loss = F.cross_entropy(
                    mixed_logits[novel_mask], query_local_labels[novel_mask]
                )
            else:
                novel_loss = mixed_logits.new_zeros(())
            route_loss = F.kl_div(
                router_probabilities.clamp_min(1e-8).log(),
                oracle_router,
                reduction="batchmean",
            )

            original_probabilities = F.softmax(
                original_logits.float().detach(), dim=-1
            )
            safe_mask = (
                original_logits.argmax(dim=-1) == query_local_labels
            )
            if torch.any(safe_mask):
                safe_loss = F.kl_div(
                    F.log_softmax(mixed_logits[safe_mask], dim=-1),
                    original_probabilities[safe_mask],
                    reduction="batchmean",
                )
            else:
                safe_loss = mixed_logits.new_zeros(())

            loss = (
                classification_loss
                + router_cfg.NOVEL_LOSS_WEIGHT * novel_loss
                + router_cfg.ROUTE_LOSS_WEIGHT * route_loss
                + router_cfg.SAFE_KL_WEIGHT * safe_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if router_cfg.GRAD_CLIP > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.frequency_router.parameters(), router_cfg.GRAD_CLIP
                )
            optimizer.step()

            with torch.no_grad():
                running_loss += float(loss.item())
                running_original_correct += int(
                    (original_logits.argmax(dim=-1) == query_local_labels)
                    .sum()
                    .item()
                )
                running_mixed_correct += int(
                    (mixed_logits.argmax(dim=-1) == query_local_labels)
                    .sum()
                    .item()
                )
                running_count += query_local_labels.numel()
                running_steps += 1
                mean_router = router_probabilities.mean(dim=0)
                if running_router is None:
                    running_router = mean_router
                else:
                    running_router = running_router + mean_router

            should_log = (
                step == 1
                or step == router_cfg.TRAIN_STEPS
                or step % router_cfg.LOG_INTERVAL == 0
            )
            if should_log:
                weights = (running_router / running_steps).tolist()
                print(
                    "router step {}/{}: loss={:.4f}, original={:.2f}%, "
                    "mixed={:.2f}%, weights(null/low/mid/high)={}".format(
                        step,
                        router_cfg.TRAIN_STEPS,
                        running_loss / running_steps,
                        100.0 * running_original_correct / running_count,
                        100.0 * running_mixed_correct / running_count,
                        ", ".join("{:.3f}".format(value) for value in weights),
                    )
                )
                running_loss = 0.0
                running_original_correct = 0
                running_mixed_correct = 0
                running_count = 0
                running_router = None
                running_steps = 0

        self.frequency_router.eval()
        print("frequency residual router training complete; shared router frozen")
    

    def build_task_statistics(self, class_names, loader,
                         class_index, calibrate_novel_vision_proto=False):
        
            
        def shrink_cov(cov, alpha1=1.0, alpha2=0.0):
            diag_mean = torch.mean(torch.diagonal(cov))
            off_diag = cov.clone()
            off_diag.fill_diagonal_(0.0)
            mask = off_diag != 0.0
            off_diag_mean = (off_diag*mask).sum() / mask.sum()
            iden = torch.eye(cov.shape[0]).to(cov.device)
            cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
            return cov_


        cls_begin_index = int(class_index[0])


        text_features, text_targets = self.inference_text_feature(class_names, self.template, cls_begin_index)

        (
            description_features,
            description_targets,
            description_proto,
            frequency_description_proto,
        ) = self.inference_all_description_feature(
            class_names=class_names,
            gpt_path=self.cfg.DATASET.GPT_PATH,
            cls_begin_index=cls_begin_index,
        )
        
        (
            images_features,
            images_targets,
            images_proto,
            frequency_image_proto,
            frequency_uncertainty,
            frequency_sample_counts,
            all_frequency_features,
            frequency_modality_descriptors,
        ) = self.inference_all_img_feature(
            loader, cls_begin_index, class_index=class_index
        )
        # Explicit prompt selection must use only the current session's raw
        # support-set prototypes, before optional novel-to-base calibration.
        frequency_grounding_proto = frequency_image_proto

        if cls_begin_index != 0:
            if calibrate_novel_vision_proto:
                print(f'calibrate vision proto on class [{class_index}]')
                images_proto = self.soft_calibration(self.base_vision_prototype, images_proto)
                if (
                    self.frequency_enabled
                    and self.cfg.TRAINER.BiMC.FREQUENCY.NOVEL_VISION_CALIBRATION
                ):
                    print('calibrate low/middle/high visual prototypes independently')
                    calibrated_bands = []
                    for band_id in range(frequency_image_proto.shape[1]):
                        calibrated_bands.append(
                            self.soft_calibration(
                                self.base_frequency_vision_prototype[:, band_id],
                                frequency_image_proto[:, band_id],
                            )
                        )
                    frequency_image_proto = torch.stack(calibrated_bands, dim=1)
        else:
            self.base_vision_prototype = images_proto
            if self.frequency_enabled:
                self.base_frequency_vision_prototype = frequency_image_proto

        frequency_state = {}
        if self.frequency_enabled:
            frequency_cfg = self.cfg.TRAINER.BiMC.FREQUENCY
            frequency_prompt_proto = self.inference_frequency_text_feature(class_names)
            if frequency_cfg.USE_EXPLICIT_DESCRIPTIONS:
                frequency_description_candidates = (
                    self.inference_explicit_frequency_description_candidates(
                        class_names,
                        frequency_cfg.EXPLICIT_DESCRIPTION_PATH,
                    )
                )
                (
                    frequency_description_proto,
                    selected_description_scores,
                    selected_description_indices,
                ) = select_frequency_description_prototypes(
                    text_candidates=frequency_description_candidates,
                    visual_prototypes=frequency_grounding_proto,
                    top_k=frequency_cfg.DESCRIPTION_TOPK,
                    temperature=frequency_cfg.DESCRIPTION_TEMPERATURE,
                )
                print(
                    'explicit frequency descriptions: top-k={}, '
                    'mean selected similarity={:.3f}'.format(
                        selected_description_indices.shape[-1],
                        selected_description_scores.float().mean().item(),
                    )
                )
            description_weight = frequency_cfg.DESCRIPTION_WEIGHT
            frequency_semantic_proto = F.normalize(
                (1.0 - description_weight) * frequency_prompt_proto
                + description_weight * frequency_description_proto,
                dim=-1,
            )
            (
                frequency_calibrated_proto,
                frequency_band_weights,
                frequency_semantic_gates,
                frequency_alignment,
            ) = calibrate_frequency_prototypes(
                visual_prototypes=frequency_image_proto,
                semantic_prototypes=frequency_semantic_proto,
                uncertainty=frequency_uncertainty,
                semantic_weight=frequency_cfg.SEMANTIC_WEIGHT,
                max_semantic_weight=frequency_cfg.MAX_SEMANTIC_WEIGHT,
                uncertainty_scale=frequency_cfg.UNCERTAINTY_SCALE,
                alignment_scale=frequency_cfg.ALIGNMENT_SCALE,
                fusion_temperature=frequency_cfg.FUSION_TEMPERATURE,
                adaptive_fusion=frequency_cfg.ADAPTIVE_FUSION,
                semantic_gate_mode=frequency_cfg.SEMANTIC_GATE_MODE,
                band_prior=torch.as_tensor(
                    frequency_cfg.BAND_PRIOR,
                    device=frequency_image_proto.device,
                    dtype=frequency_image_proto.dtype,
                ),
            )
            if frequency_cfg.RELIABILITY_ALPHA:
                frequency_class_alpha, frequency_reliability = (
                    compute_frequency_class_alpha(
                        alignment=frequency_alignment,
                        uncertainty=frequency_uncertainty,
                        band_weights=frequency_band_weights,
                        sample_counts=frequency_sample_counts,
                        max_alpha=frequency_cfg.FREQ_ALPHA,
                        min_alpha=frequency_cfg.MIN_FREQ_ALPHA,
                        uncertainty_scale=(
                            frequency_cfg.RELIABILITY_UNCERTAINTY_SCALE
                        ),
                        shot_tau=frequency_cfg.RELIABILITY_SHOT_TAU,
                        reliability_power=frequency_cfg.RELIABILITY_POWER,
                    )
                )
            else:
                frequency_class_alpha = frequency_image_proto.new_full(
                    (frequency_image_proto.shape[0],), frequency_cfg.FREQ_ALPHA
                )
                frequency_reliability = frequency_image_proto.new_ones(
                    frequency_image_proto.shape[0]
                )
            mean_weights = frequency_band_weights.float().mean(dim=0).tolist()
            print(
                'mean frequency weights (low/middle/high): '
                + ', '.join('{:.3f}'.format(value) for value in mean_weights)
            )
            print(
                'frequency alpha: mean={:.3f}, min={:.3f}, max={:.3f}'.format(
                    frequency_class_alpha.float().mean().item(),
                    frequency_class_alpha.float().min().item(),
                    frequency_class_alpha.float().max().item(),
                )
            )
            frequency_state = {
                'frequency_image_proto': frequency_image_proto,
                'frequency_prompt_proto': frequency_prompt_proto,
                'frequency_description_proto': frequency_description_proto,
                'frequency_semantic_proto': frequency_semantic_proto,
                'frequency_calibrated_proto': frequency_calibrated_proto,
                'frequency_uncertainty': frequency_uncertainty,
                'frequency_sample_counts': frequency_sample_counts,
                'frequency_band_weights': frequency_band_weights,
                'frequency_semantic_gates': frequency_semantic_gates,
                'frequency_alignment': frequency_alignment,
                'frequency_class_alpha': frequency_class_alpha,
                'frequency_reliability': frequency_reliability,
            }
            if cls_begin_index == 0 and self.frequency_router is not None:
                self.fit_frequency_router(
                    image_features=images_features,
                    frequency_features=all_frequency_features,
                    labels=images_targets,
                    class_index=class_index,
                    text_features=text_features,
                    description_proto=description_proto,
                    frequency_semantic_proto=frequency_semantic_proto,
                )

        frequency_modality_state = {}
        if self.frequency_modality_enabled:
            modality_cfg = self.cfg.TRAINER.BiMC.FREQUENCY_MODALITY
            if cls_begin_index == 0:
                self.fit_frequency_modality_encoder(
                    descriptors=frequency_modality_descriptors,
                    image_features=images_features,
                    labels=images_targets,
                    class_index=class_index,
                    text_features=text_features,
                )
            if not self.frequency_modality_fitted:
                raise RuntimeError(
                    "The frequency modality must be fitted on the base session first."
                )
            with torch.no_grad():
                frequency_modality_features = self.frequency_modality_encoder(
                    frequency_modality_descriptors
                )
                (
                    frequency_modality_proto,
                    frequency_modality_uncertainty,
                    frequency_modality_sample_counts,
                ) = compute_modality_prototypes(
                    frequency_modality_features,
                    images_targets,
                    class_index,
                )

            if cls_begin_index == 0:
                self.base_frequency_modality_prototype = frequency_modality_proto
            elif modality_cfg.NOVEL_CALIBRATION:
                print("calibrate novel frequency-modality prototypes")
                frequency_modality_proto = self.soft_calibration(
                    self.base_frequency_modality_prototype,
                    frequency_modality_proto,
                )

            if modality_cfg.RELIABILITY_ALPHA:
                (
                    frequency_modality_alpha,
                    frequency_modality_reliability,
                ) = compute_modality_alpha(
                    uncertainty=frequency_modality_uncertainty,
                    sample_counts=frequency_modality_sample_counts,
                    max_alpha=modality_cfg.FUSION_WEIGHT,
                    min_alpha=modality_cfg.MIN_FUSION_WEIGHT,
                    uncertainty_scale=modality_cfg.UNCERTAINTY_SCALE,
                    shot_tau=modality_cfg.SHOT_TAU,
                )
            else:
                frequency_modality_alpha = frequency_modality_proto.new_full(
                    (frequency_modality_proto.shape[0],),
                    modality_cfg.FUSION_WEIGHT,
                )
                frequency_modality_reliability = frequency_modality_proto.new_ones(
                    frequency_modality_proto.shape[0]
                )
            print(
                "frequency modality alpha: mean={:.3f}, min={:.3f}, max={:.3f}; "
                "uncertainty={:.3f}".format(
                    frequency_modality_alpha.mean().item(),
                    frequency_modality_alpha.min().item(),
                    frequency_modality_alpha.max().item(),
                    frequency_modality_uncertainty.mean().item(),
                )
            )
            frequency_modality_state = {
                'frequency_modality_proto': frequency_modality_proto,
                'frequency_modality_uncertainty': frequency_modality_uncertainty,
                'frequency_modality_sample_counts': frequency_modality_sample_counts,
                'frequency_modality_alpha': frequency_modality_alpha,
                'frequency_modality_reliability': frequency_modality_reliability,
            }


        cov_images = torch.cov(images_features.T)

        if cls_begin_index == 0:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_BASE) 
        else:
            cov_images = shrink_cov(cov_images, alpha1=self.cfg.TRAINER.BiMC.GAMMA_INC)

        
        print('finish loading covariance')

        state = {
            'description_proto': description_proto,
            'description_features': description_features,
            'description_targets': description_targets,

            'text_features': text_features,
            'text_targets': text_targets,           
  
            'image_proto': images_proto,
            'images_features': images_features,
            'images_targets': images_targets,
            'cov_image': cov_images,
            
            'class_index': class_index,
            'sample_cnt': len(images_features)
        }
        state.update(frequency_state)
        state.update(frequency_modality_state)
        return state

   

    def forward_ours(self, images, num_cls, num_base_cls,
                           image_proto, cov_image,
                           description_proto,
                           description_features, description_targets,
                           text_features,
                           beta,
                           frequency_proto=None,
                           frequency_band_weights=None,
                           frequency_class_alpha=None,
                           frequency_modality_proto=None,
                           frequency_modality_alpha=None):
    
        def knn_similarity_scores(queries, support_features, support_labels):
            """
            Compute the similarity between each query sample and all support samples,
            and retrieve the maximum score for each class per query.
            """
            # Ensure all inputs are on the same device
            device = queries.device
            support_features = support_features.to(device)
            support_labels = support_labels.to(device)
            similarity_scores = torch.matmul(queries, support_features.T)
            k = torch.max(support_labels) + 1
            max_scores = torch.full((queries.size(0), k), float('-inf'), device=device)
            expanded_labels = support_labels.unsqueeze(0).expand(queries.size(0), -1)
            for label in range(k):
                label_mask = (expanded_labels == label)
                masked_scores = similarity_scores.masked_fill(~label_mask, float('-inf'))
                max_scores[:, label] = torch.max(masked_scores, dim=1).values
            return max_scores


        def _mahalanobis(dist, cov_inv):
            """
            Compute the Mahalanobis distance between feature vectors and a class prototype.
            """
            left_term = torch.matmul(dist, cov_inv)
            mahal = torch.matmul(left_term, dist.T)
            return torch.diag(mahal)


        def _cov_forward(feat, proto, cov):
            """
            Perform a forward pass computing negative Mahalanobis distance between 
            features and each class prototype using a shared covariance matrix.
            """
            maha_dist = []
            inv_covmat = torch.pinverse(cov.to(dtype=torch.float32))
            inv_covmat = inv_covmat.to(dtype=proto.dtype)
            for cl in range(num_cls):
                distance = feat - proto[cl]
                dist = _mahalanobis(distance, inv_covmat)
                maha_dist.append(dist)
            maha_dist = torch.stack(maha_dist)
            logits = -maha_dist.T
            return logits
        

        # Normalize the image features
        img_feat = self.extract_img_feature(images)
        img_feat = F.normalize(img_feat, dim=-1)

        if self.cfg.TRAINER.BiMC.TEXT_CALIBRATION:
            lambda_t = self.cfg.TRAINER.BiMC.LAMBDA_T
        else:
            lambda_t = 0.0

        # Here we compute the classifier after modality calibration. 
        # Note that image_proto has already been calibrated in the `build_task_statistics` function.
        fused_proto = beta * ((1 - lambda_t) * text_features + lambda_t * description_proto) + (1 - beta) * image_proto        
        fused_proto = F.normalize(fused_proto, dim=-1)  
        logits_proto_fused = img_feat @ fused_proto.t()
        prob_fused_proto = F.softmax(logits_proto_fused, dim=-1)

        if self.frequency_enabled:
            if frequency_proto is None or frequency_band_weights is None:
                raise ValueError(
                    "Frequency mode requires calibrated prototypes and band weights."
                )
            frequency_features = self.extract_frequency_img_feature(images)
            if self.frequency_router is not None:
                band_logits = compute_frequency_band_logits(
                    frequency_features, frequency_proto
                )
                router_probabilities = self.frequency_router(
                    logits_proto_fused, band_logits
                )
                router_cfg = self.cfg.TRAINER.BiMC.FREQUENCY.ROUTER
                router_class_alpha = (
                    frequency_class_alpha
                    if router_cfg.USE_CLASS_ALPHA
                    else None
                )
                routed_logits = residual_frequency_fusion(
                    original_logits=logits_proto_fused,
                    band_logits=band_logits,
                    router_probabilities=router_probabilities,
                    class_band_weights=frequency_band_weights,
                    max_alpha=router_cfg.MAX_ALPHA,
                    class_alpha=router_class_alpha,
                )
                prob_fused_proto = F.softmax(routed_logits, dim=-1)
            else:
                logits_frequency = compute_frequency_logits(
                    frequency_features,
                    frequency_proto,
                    frequency_band_weights,
                )
                prob_frequency = F.softmax(logits_frequency, dim=-1)
                if frequency_class_alpha is None:
                    frequency_class_alpha = prob_frequency.new_full(
                        (prob_frequency.shape[1],),
                        self.cfg.TRAINER.BiMC.FREQUENCY.FREQ_ALPHA,
                    )
                prob_fused_proto = mix_frequency_probabilities(
                    prob_fused_proto,
                    prob_frequency,
                    frequency_class_alpha,
                )

        if self.frequency_modality_enabled:
            if frequency_modality_proto is None:
                raise ValueError(
                    "Tri-modal mode requires frequency_modality_proto."
                )
            modality_cfg = self.cfg.TRAINER.BiMC.FREQUENCY_MODALITY
            frequency_modality_features = self.extract_frequency_modality_feature(
                images
            )
            frequency_modality_logits = (
                frequency_modality_features.float()
                @ F.normalize(frequency_modality_proto.float(), dim=-1).t()
            ) / modality_cfg.TEMPERATURE
            frequency_modality_prob = F.softmax(
                frequency_modality_logits, dim=-1
            )
            if frequency_modality_alpha is None:
                frequency_modality_alpha = frequency_modality_prob.new_full(
                    (frequency_modality_prob.shape[1],),
                    modality_cfg.FUSION_WEIGHT,
                )
            prob_fused_proto = fuse_modality_probabilities(
                prob_fused_proto.float(),
                frequency_modality_prob,
                frequency_modality_alpha,
            )

        logits_cov = _cov_forward(img_feat, image_proto, cov_image)
        logits_knn = knn_similarity_scores(img_feat, description_features, description_targets)    
        prob_cov = F.softmax(logits_cov / 512, dim=-1)
        prob_knn = F.softmax(logits_knn, dim=-1)

        NUM_BASE_CLS = num_base_cls
        use_diversity = self.cfg.TRAINER.BiMC.USING_ENSEMBLE
        if use_diversity:
            ensemble_alpha = self.cfg.DATASET.ENSEMBLE_ALPHA
        else:
            ensemble_alpha = 1.0

        base_probs = ensemble_alpha * prob_fused_proto[:, :NUM_BASE_CLS] + (1 - ensemble_alpha) * prob_cov[:, :NUM_BASE_CLS]
        inc_probs = ensemble_alpha * prob_fused_proto[:, NUM_BASE_CLS:] + (1 - ensemble_alpha) * prob_knn[:, NUM_BASE_CLS:]

        prob_fused = torch.cat([base_probs, inc_probs], dim=1)
        logits = prob_fused
        return logits



    @torch.no_grad()
    def extract_img_feature(self, images):
        images = images.to(self.device)
        image_features = self.clip_model.encode_image(images)
        return image_features


    @torch.no_grad()
    def extract_frequency_img_feature(self, images):
        """Encode low/middle/high filtered images with the frozen CLIP model."""
        if not self.frequency_enabled or self.frequency_decomposer is None:
            raise RuntimeError("Frequency feature extraction is not enabled.")
        images = images.to(self.device)
        if self.frequency_decomposer.view_mode == "original":
            features = F.normalize(self.clip_model.encode_image(images), dim=-1)
            return features.unsqueeze(1).expand(-1, 3, -1)
        band_images = self.frequency_decomposer(images)
        band_features = []
        # Encode each band separately to keep peak memory close to baseline.
        for band_id in range(band_images.shape[1]):
            features = self.clip_model.encode_image(band_images[:, band_id])
            band_features.append(F.normalize(features, dim=-1))
        return torch.stack(band_features, dim=1)


    @torch.no_grad()
    def extract_frequency_modality_descriptor(self, images):
        """Compute phase-free FFT descriptors without using CLIP."""
        if (
            not self.frequency_modality_enabled
            or self.frequency_modality_descriptor is None
        ):
            raise RuntimeError("Independent frequency modality is not enabled.")
        return self.frequency_modality_descriptor(images.to(self.device))


    @torch.no_grad()
    def extract_frequency_modality_feature(self, images):
        """Encode an image through the frozen independent frequency tower."""
        if not self.frequency_modality_fitted:
            raise RuntimeError(
                "Independent frequency encoder must be fitted on the base session."
            )
        descriptors = self.extract_frequency_modality_descriptor(images)
        return self.frequency_modality_encoder(descriptors)


    @torch.no_grad()
    def forward(self, images):
        img_feat = self.extract_img_feature(images)
        img_feat = F.normalize(img_feat, dim=-1)
        classifier = F.normalize(self.classifier_weights, dim=-1)
        logits = 100. * img_feat @ classifier.t()
        return logits



    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets
