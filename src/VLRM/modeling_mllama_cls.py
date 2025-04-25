from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.models.mllama.configuration_mllama import MllamaConfig
from transformers.models.mllama.modeling_mllama import (
    MllamaForCausalLM, MllamaPreTrainedModel, MllamaVisionModel,
    _prepare_cross_attention_mask)
from transformers.utils import logging

logger = logging.get_logger(__name__)

class MllamaForSequenceClassification(MllamaPreTrainedModel, GenerationMixin):
    _supports_quantized_cache = False  # quant cache not supported in encoder-decoder setting

    def __init__(self, config: MllamaConfig):
        super().__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.hidden_size = config.text_config.hidden_size
        self.max_num_tiles = config.vision_config.max_num_tiles
        self.vision_output_dim = config.vision_config.vision_output_dim
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

        self.vision_model = MllamaVisionModel._from_config(config.vision_config)
        self.language_model = MllamaForCausalLM._from_config(config.text_config)
        self.multi_modal_projector = nn.Linear(
            config.vision_config.vision_output_dim,
            config.text_config.hidden_size,
            bias=True,
        )
        self.num_labels = config.num_labels
        self.score = nn.Linear(self.hidden_size, self.num_labels, bias=False)
        
        self.post_init()

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        aspect_ratio_mask: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and cross_attention_states is not None:
            raise ValueError("`pixel_values` and `cross_attention_states` cannot be provided simultaneously")

        if pixel_values is not None:
            if aspect_ratio_ids is None:
                raise ValueError("`aspect_ratio_ids` must be provided if `pixel_values` is provided")
            # get vision tokens from vision model
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            cross_attention_states = vision_outputs[0]
            cross_attention_states = self.multi_modal_projector(cross_attention_states).reshape(
                -1, cross_attention_states.shape[-2], self.hidden_size
            )

        if cross_attention_mask is not None:
            cross_attention_mask, full_text_row_masked_out_mask = _prepare_cross_attention_mask(
                cross_attention_mask,
                num_vision_tokens=self.vision_model.num_patches,
                dtype=self.dtype,
            )
        else:
            full_text_row_masked_out_mask = None

        if cross_attention_mask is not None and cache_position is not None:
            cross_attention_mask = cross_attention_mask[:, :, cache_position]
            full_text_row_masked_out_mask = full_text_row_masked_out_mask[:, :, cache_position]
            
        transformer_outputs = self.language_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        aspect_ratio_ids=None,
        aspect_ratio_mask=None,
        cross_attention_mask=None,
        past_key_values=None,
        use_cache=False,
        cache_position=None,
        num_logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # TODO: we have no attention_mask so this won't work, check if we really won't need attention mask and find another way
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cross_attention_mask": cross_attention_mask,
            }
        )

        # If we're in pre-fill or cacheless decoding step, then we need pixel_values and aspect ratios
        # to compute image hidden states, otherwise they are cached within each cross attn layer
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["aspect_ratio_ids"] = aspect_ratio_ids
            model_inputs["aspect_ratio_mask"] = aspect_ratio_mask

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        cross_attention_mask_prev = model_kwargs.get("cross_attention_mask", None)
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )

        # add cross-attn mask for new token
        if cross_attention_mask_prev is not None:
            model_kwargs["cross_attention_mask"] = torch.cat(
                [cross_attention_mask_prev, cross_attention_mask_prev[:, -1:, ...]], dim=1
            )
        return model_kwargs
