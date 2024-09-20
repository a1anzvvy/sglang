import inspect
import os
import os.path as osp
from typing import Any, Dict, Iterable, Optional, Tuple, List, Union
import numpy as np

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    LlavaConfig,
    LlamaConfig,
    PretrainedConfig,
    PreTrainedModel,
    SiglipVisionModel,
)
from vllm.config import CacheConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .srgpt_dep.mm_projector import build_mm_projector
from .srgpt_dep.region_extractor import build_region_extractor

from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.managers.schedule_batch import ImageInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
LLM_MASK_TOKEN_INDEX = 128257
IMAGE_TOKEN_LENGTH = 196
PAD_TOKEN_ID = 128256

class VilaLlavaLlamaConfig(LlavaConfig):
    model_type = "vila_llava_llama"

class VilaLlavaLlamaModel(nn.Module):
    def __init__(
        self,
        config: LlavaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.root_path = config._name_or_path
        text_config = LlamaConfig.from_dict(self.config.text_config.to_dict())
        self.language_model = LlamaForCausalLM(text_config, quant_config=quant_config)

    def pad_input_ids(
        self,
        input_ids: List[int],
        image_inputs: ImageInputs,
    ):
        image_sizes, pad_values = image_inputs.image_sizes, image_inputs.pad_values

        if IMAGE_TOKEN_INDEX not in input_ids:
            return input_ids, [0]
        
        #  FIXME: Note index only return the first index, to support multi-image need fix this
        offset = input_ids.index(IMAGE_TOKEN_INDEX)

        # Make sure the length is IMAGE_TOKEN_LENGTH
        pad_ids = pad_values * (
            IMAGE_TOKEN_LENGTH // len(pad_values)
        )
        new_input_ids = (
            input_ids[:offset]
            + pad_ids
            + input_ids[(offset+1):]
        )

        return new_input_ids

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        mm_projector_path = os.path.join(self.root_path, "mm_projector")
        self.multi_modal_projector = build_mm_projector(mm_projector_path, self.config).cuda()
        self.multi_modal_projector.eval()

        region_extrator_path = os.path.join(self.root_path,"region_extractor")
        self.region_extractor = build_region_extractor(region_extrator_path, self.config).cuda()
        self.region_extractor.eval()

        vision_path = os.path.join(self.root_path, "vision_tower")
        self.vision_tower = SiglipVisionModel.from_pretrained(vision_path, torch_dtype=torch.bfloat16).cuda()
        self.vision_tower.eval()

        self.vision_feature_layer = self.config.mm_vision_select_layer
        self.vision_feature_select_strategy = self.config.mm_vision_select_feature
        self.image_size = self.vision_tower.config.image_size
        self.patch_size = self.vision_tower.config.patch_size

        self.image_aspect_ratio = getattr(self.config, "image_aspect_ratio")

        self.image_feature_len = int((self.image_size // self.patch_size) ** 2)
        if (
            self.vision_feature_select_strategy == "patch"
            or self.vision_feature_select_strategy == "full"
        ):
            pass
        elif self.vision_feature_select_strategy == "cls_patch":
            self.image_feature_len += 1
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        
        for name, loaded_weight in weights:
            self.language_model.load_weights([(name, loaded_weight)])

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        image_inputs = forward_batch.image_inputs

        if forward_batch.forward_mode.is_extend():
            bs = forward_batch.batch_size

            modalities_list = []
            max_image_offset = []
            for im in image_inputs:
                if im and im.modalities is not None:
                    modalities_list.extend(im.modalities)
                if im and im.image_offsets is not None:
                    max_image_offset.append(max(im.image_offsets))
                else:
                    max_image_offset.append(-1)

            # Embed text. Vision tokens will be replaced by image features later.
            input_embeds = self.language_model.model.embed_tokens(input_ids)

            # Whether the requests need vision inputs
            start_positions = positions[forward_batch.extend_start_loc].cpu().numpy()
            need_vision = start_positions <= max_image_offset

            if need_vision.any():
                pixel_values = [
                    image_inputs[i].pixel_values for i in range(bs) if need_vision[i]
                ]
                image_sizes = [
                    image_inputs[i].image_sizes for i in range(bs) if need_vision[i]
                ]
                image_offsets = [
                    image_inputs[i].image_offsets for i in range(bs) if need_vision[i]
                ]
                region_coords = [
                    image_inputs[i].region_coords for i in range(bs) if need_vision[i]
                ]

                ########## Encode Image ########

                if pixel_values[0].ndim == 4:
                    # Video: BS, num_images, C=3, H=336, W=336, num_images obtained from process_images
                    np.concatenate(pixel_values, axis=0)
                    # ndim=4
                    concat_images = torch.tensor(
                        np.concatenate(pixel_values, axis=0),
                        device=self.vision_tower.device,
                    )
                    tower_features = self.vision_tower(concat_images).last_hidden_state
                    split_sizes = [image.shape[0] for image in pixel_values]
                    tower_features = torch.split(tower_features, split_sizes, dim=0)
                    hres_tower_features, lres_tower_features = self.region_extractor.feature_refinement(tower_features)
                    image_features = self.multi_modal_projector(lres_tower_features)
                    # hd image_features: BS, num_images, h, w
                else:
                    # Image: normal pixel: BS, C=3, H=336, W=336
                    pixel_values = torch.tensor(
                        pixel_values, device=self.vision_tower.device, dtype=torch.bfloat16,
                    )
                    tower_features = self.vision_tower(pixel_values, output_hidden_states=True).hidden_states[-2]
                    hres_tower_features, lres_tower_features = self.region_extractor.feature_refinement(tower_features)
                    image_features = self.multi_modal_projector(lres_tower_features)
                    image_features = image_features.unsqueeze(1)
                    # image_features: BS, 1, h, w

                ########## Fill Image Features ########
                extend_start_loc_cpu = forward_batch.extend_start_loc.cpu().numpy()
                prefix_lens_cpu = forward_batch.extend_prefix_lens.cpu().numpy()

                pt = 0
                for i in range(bs):
                    if not need_vision[i]:
                        continue

                    start_idx = extend_start_loc_cpu[i]
                    prefix_len = prefix_lens_cpu[i]

                    # Multiple images
                    for j, image_offset in enumerate(image_offsets[i]):
                        if image_offset < prefix_len:
                            continue

                        tmp_image_feature = image_features[pt][j]
                        left_idx = start_idx + (image_offset - prefix_len)
                        right_idx = start_idx + (image_offset - prefix_len) + IMAGE_TOKEN_LENGTH

                        try:
                            input_embeds[left_idx:right_idx] = tmp_image_feature
                        except RuntimeError as e:
                            print(f"RuntimeError in image encoding: {e}")
                            print(f"{input_embeds.shape=}, {tmp_image_feature.shape=}")
                            print(
                                f"{start_idx=}, {image_offset=}, {prefix_len=}"
                            )

                    pt+=1

                ########## Extract Region and Fill Mask Featrues ########
                mask_indexs = torch.nonzero(input_ids == LLM_MASK_TOKEN_INDEX, as_tuple=True)[0]
                # FIXME: This assumes each image has only one or none <mask> token
                assert len(mask_indexs) <= sum([len(image_offsets[i]) for i in range(bs)])
                assert sum([len(l) for l in region_coords]) == mask_indexs.shape[0]
                for i in range(len(mask_indexs)):
                    region_mask = torch.zeros((1, 1, self.image_size, self.image_size), dtype=torch.float16, device=self.vision_tower.device)
                    region_coord = region_coords[i]
                    # FIXME: assuming there is only one mask in one image
                    region_idx = 0
                    region_mask[0,
                                0,
                                region_coord[region_idx][2]:region_coord[region_idx][3],
                                region_coord[region_idx][0]:region_coord[region_idx][1]] = 1
                    # FIXME: index of mask should match index of images
                    mask_embed, _ = self.region_extractor(hres_tower_features[i:i+1], None, region_mask)
                    mask_embed = mask_embed[0]
                    replace_mask_idx = mask_indexs[i].cpu().numpy()
                    # FIXME: This assumes each image has only one <mask> token
                    if replace_mask_idx >= prefix_len:
                        input_embeds[replace_mask_idx] = mask_embed

                        # pos_mask = (input_ids == LLM_MASK_TOKEN_INDEX)
                        # zeros_embeds = torch.zeros_like(input_embeds)
                        # zeros_embeds[pos_mask] = mask_embed.to(device=zeros_embeds.device, dtype=zeros_embeds.dtype)
                        # input_embeds = input_embeds * (~pos_mask).to(input_embeds.dtype).unsqueeze(-1) + zeros_embeds

            return self.language_model(input_ids, positions, forward_batch, input_embeds=input_embeds,)
        elif forward_batch.forward_mode.is_decode():
            return self.language_model(input_ids, positions, forward_batch)

EntryClass = [VilaLlavaLlamaModel]