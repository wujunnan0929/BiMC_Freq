import torch
import torch.nn as nn
import torch.nn.functional as F
import models.clip.clip as clip
import json

from models.frequency import (
    RadialFrequencyDecomposer,
    calibrate_frequency_prototypes,
    compute_frequency_class_alpha,
    compute_frequency_logits,
    compute_frequency_prototypes,
    mix_frequency_probabilities,
    route_description_embeddings,
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
            self.frequency_decomposer = RadialFrequencyDecomposer(
                low_cutoff=frequency_cfg.LOW_CUTOFF,
                high_cutoff=frequency_cfg.HIGH_CUTOFF,
                center_residual_bands=frequency_cfg.CENTER_RESIDUAL_BANDS,
                fft_batch_size=frequency_cfg.FFT_BATCH_SIZE,
                fft_device=frequency_cfg.FFT_DEVICE,
                view_mode=frequency_cfg.VIEW_MODE,
                high_enhance=frequency_cfg.HIGH_ENHANCE,
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


    @torch.no_grad()
    def inference_all_img_feature(self, loader, cls_begin_index, class_index=None):
        all_features = []
        all_labels = []
        all_frequency_features = []
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

        if not self.frequency_enabled:
            return all_features, all_labels, prototypes, None, None, None

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
            if self.frequency_enabled:
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
        if self.frequency_enabled:
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
        ) = self.inference_all_img_feature(
            loader, cls_begin_index, class_index=class_index
        )

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
        return state

   

    def forward_ours(self, images, num_cls, num_base_cls,
                           image_proto, cov_image,
                           description_proto,
                           description_features, description_targets,
                           text_features,
                           beta,
                           frequency_proto=None,
                           frequency_band_weights=None,
                           frequency_class_alpha=None):
    
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
