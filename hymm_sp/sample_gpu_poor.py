import os
import numpy as np
import torch
from einops import rearrange
import imageio
from .sample_inference_audio import HunyuanVideoSampler
from .data_kits.face_align import AlignImage
from transformers import WhisperModel
from transformers import AutoFeatureExtractor
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

def audio_image_load(base_dir, device):

    # load audio model
    wav2vec = WhisperModel.from_pretrained(os.path.join(base_dir,"whisper-tiny")).to(device=device, dtype=torch.float32)
    wav2vec.requires_grad_(False)

    feature_extractor = AutoFeatureExtractor.from_pretrained(os.path.join(base_dir,"whisper-tiny"))

    # load face align model
    align_instance = AlignImage("cuda", det_path=os.path.join(base_dir, 'det_align/detface.pt'))

    return wav2vec, feature_extractor, align_instance


def tranformer_load(args):
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args)
    args = hunyuan_video_sampler.args
    if args.cpu_offload:
        from diffusers.hooks import apply_group_offloading
        onload_device = torch.device("cuda")
        apply_group_offloading(hunyuan_video_sampler.pipeline.transformer, onload_device=onload_device, offload_type="block_level", num_blocks_per_group=1)
    return hunyuan_video_sampler

def encode_prompt_audio_text_base(
    prompt, 
    uncond_prompt, 
    pixel_value_llava, 
    uncond_pixel_value_llava, 
    device,
    num_images_per_prompt,
    do_classifier_free_guidance,
    negative_prompt=None,
    prompt_embeds = None,
    negative_prompt_embeds= None,
    lora_scale = None,
    clip_skip = None,
    text_encoder = None,
    data_type = "image",
):
    # if text_encoder is None:
    #     text_encoder = self.text_encoder

    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    # if lora_scale is not None and isinstance(self, LoraLoaderMixin):
    #     self._lora_scale = lora_scale

    #     # dynamically adjust the LoRA scale
    #     if not USE_PEFT_BACKEND:
    #         adjust_lora_scale_text_encoder(text_encoder.model, lora_scale)
    #     else:
    #         scale_lora_layers(text_encoder.model, lora_scale)

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
        
    prompt_embeds = None
    
    if prompt_embeds is None:
        # textual inversion: process multi-vector tokens if necessary
        # if isinstance(self, TextualInversionLoaderMixin):
        #     prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)
        text_inputs = text_encoder.text2tokens(prompt, data_type=data_type) # data_type: video, text_inputs: {'input_ids', 'attention_mask'}
        
        text_keys = ['input_ids', 'attention_mask']
        
        if pixel_value_llava is not None:
            text_inputs['pixel_value_llava'] = pixel_value_llava
            text_inputs['attention_mask'] = torch.cat([text_inputs['attention_mask'], torch.ones((1, 575)).to(text_inputs['attention_mask'])], dim=1)

    
        if clip_skip is None:
            prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)
            prompt_embeds = prompt_outputs.hidden_state
        else:
            prompt_outputs = text_encoder.encode(text_inputs, output_hidden_states=True, data_type=data_type)
            # Access the `hidden_states` first, that contains a tuple of
            # all the hidden states from the encoder layers. Then index into
            # the tuple to access the hidden states from the desired layer.
            prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
            # We also need to apply the final LayerNorm here to not mess with the
            # representations. The `last_hidden_states` that we typically use for
            # obtaining the final prompt representations passes through the LayerNorm
            # layer.
            prompt_embeds = text_encoder.model.text_model.final_layer_norm(prompt_embeds)

        attention_mask = prompt_outputs.attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            bs_embed, seq_len = attention_mask.shape
            attention_mask = attention_mask.repeat(1, num_images_per_prompt)
            attention_mask = attention_mask.view(bs_embed * num_images_per_prompt, seq_len)

    if text_encoder is not None:
        prompt_embeds_dtype = text_encoder.dtype
    # elif self.unet is not None:
    #     prompt_embeds_dtype = self.unet.dtype
    # else:
    prompt_embeds_dtype = prompt_embeds.dtype

    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    if prompt_embeds.ndim == 2:
        bs_embed, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, -1)
    else:
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        # textual inversion: process multi-vector tokens if necessary
        # if isinstance(self, TextualInversionLoaderMixin):
        #     uncond_tokens = self.maybe_convert_prompt(uncond_tokens, text_encoder.tokenizer)            
        # max_length = prompt_embeds.shape[1]
        uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)

        # if hasattr(text_encoder.model.config, "use_attention_mask") and text_encoder.model.config.use_attention_mask:
        #     attention_mask = uncond_input.attention_mask.to(device)
        # else:
        #     attention_mask = None
        if uncond_pixel_value_llava is not None:
            uncond_input['pixel_value_llava'] = uncond_pixel_value_llava
            uncond_input['attention_mask'] = torch.cat([uncond_input['attention_mask'], torch.ones((1, 575)).to(uncond_input['attention_mask'])], dim=1)

        negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type)
        negative_prompt_embeds = negative_prompt_outputs.hidden_state

        negative_attention_mask = negative_prompt_outputs.attention_mask
        if negative_attention_mask is not None:
            negative_attention_mask = negative_attention_mask.to(device)
            _, seq_len = negative_attention_mask.shape
            negative_attention_mask = negative_attention_mask.repeat(1, num_images_per_prompt)
            negative_attention_mask = negative_attention_mask.view(batch_size * num_images_per_prompt, seq_len)

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if negative_prompt_embeds.ndim == 2:
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        else:
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    # if text_encoder is not None:
    #     if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
    #         # Retrieve the original scale by scaling back the LoRA layers
    #         unscale_lora_layers(text_encoder.model, lora_scale)

    return prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask

def hunyuan_avatar_main(args,hunyuan_video_sampler,json_loader,emb_data,infer_min):
    save_path=args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    frame_list=[]
    for  batch,emb_dict in zip(json_loader,emb_data):

        # fps = batch["fps"]
        # videoid = batch['videoid'][0]
        # audio_path = str(batch["audio_path"][0])
        # #save_path = args.save_path 
        # output_path = f"{save_path}/{videoid}.mp4"
        # output_audio_path = f"{save_path}/{videoid}_audio.mp4"

        if infer_min:
            batch["audio_len"][0] = 129
            
        samples = hunyuan_video_sampler.predict(args, batch,emb_dict)
       
        sample = samples['samples'][0].unsqueeze(0)  # denoised latent, (bs, 16, t//4, h//8, w//8)
        sample = sample[:, :, :batch["audio_len"][0]]
        
        video = rearrange(sample[0], "c f h w -> f h w c")
        video = (video * 255.).data.cpu().numpy().astype(np.uint8)  # （f h w c)
        print("原始 video 形状:", video.shape)  # 应为 (frames, h, w, c)
        torch.cuda.empty_cache()

        final_frames = []
        for frame in video:
            final_frames.append(frame)
        final_frames = np.stack(final_frames, axis=0)
        print("final_frames 形状:", final_frames.shape)  # 预期 (frames, h, w, c)
        frame_list.append(final_frames)
        #if rank == 0: 
        # imageio.mimsave(output_path, final_frames, fps=fps.item())
        # os.system(f"ffmpeg -i '{output_path}' -i '{audio_path}' -shortest '{output_audio_path}' -y -loglevel quiet; rm '{output_path}'")
    return frame_list[0]



    
    
    
    
    
