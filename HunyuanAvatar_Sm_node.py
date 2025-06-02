# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import random
import io
import torchaudio
from loguru import logger
from einops import rearrange
from omegaconf import OmegaConf

from .hymm_sp.sample_gpu_poor import hunyuan_avatar_main,tranformer_load,audio_image_load,encode_prompt_audio_text_base
from .node_utils import tensor_to_pil,gc_clear,load_images
from .hymm_sp.data_kits.audio_preprocessor import encode_audio, get_facemask
from .hymm_sp.text_encoder import TextEncoder
from .hymm_sp.constants import PROMPT_TEMPLATE
from .hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal

import folder_paths

MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))


device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

# add checkpoints dir
Hunyuan_Avatar_Weigths_Path = os.path.join(folder_paths.models_dir, "HunyuanAvatar")
if not os.path.exists(Hunyuan_Avatar_Weigths_Path):
    os.makedirs(Hunyuan_Avatar_Weigths_Path)
folder_paths.add_model_folder_path("HunyuanAvatar", Hunyuan_Avatar_Weigths_Path)


class HY_Avatar_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transformer": (["none"] + [i for i in folder_paths.get_filename_list("HunyuanAvatar") if i.endswith(".pt")],),
                "use_fp8":  ("BOOLEAN", {"default": True},),
                "cpu_offload":  ("BOOLEAN", {"default": True},),
               
                
            },
        }

    RETURN_TYPES = ("MODEL_HY_AVATAR_MODEL","HY_AVATAR_MODEL_ARGS")
    RETURN_NAMES = ("model","args")
    FUNCTION = "loader_main"
    CATEGORY = "HunyuanAvatar_Sm"

    def loader_main(self, transformer,use_fp8,cpu_offload,):
        vae_str="884-16c-hy0801"
        vae_channels = int(vae_str.split("-")[1][:-1])
        load_key=["module", "ema"]
        args_dict={
            "ckpt": "",
            "model":"HYVideo-T/2",
            "video_size": 512,
            "load_key":load_key[0],
            "sample_n_frames": 129,#"How many frames to sample from a video. if using 3d vae, the number should be 4n+1"
            "seed": 128,
            "image_size": 704,
            "cfg_scale": 7.5,
            "ip_cfg_scale": 0,
            "infer_steps": 50,
            "use_deepcache": 1,
            "flow_shift_eval_video": 5.0,
            "use_linear_quadratic_schedule": True,
            "use_attention_mask": True,
            "linear_schedule_end": 25.0,
            "flow_solver": "euler",
            "flow_reverse": True,
            "save_path": folder_paths.get_output_directory(),
            "use_fp8": use_fp8,
            "cpu_offload": cpu_offload,
            "infer_min": True,
            "prompt_template_video": "default",
            "precision": "fp16",#bf16
            "reproduce": True,
            "num_images": 1,
            "val_disable_autocast": True,
            "pos_prompt": "",
            "neg_prompt": "",
            "save_path_suffix": "",
            "pad_face_size": 0.7,
            "item_name":"Hunyuan_Avatar",
            "use_deepcache": 1,
            "latent_channels":vae_channels,
            "rope_theta":256,
            "vae": "884-16c-hy0801",
            "vae_tiling": True,
            "vae_precision": "fp16",
            "text_encoder":"llava-llama-3-8b",
            "tokenizer":"llava-llama-3-8b",
            "text_encoder_precision": "fp16",
            "text_states_dim": 4096,
            "text_len": 256,
            "text_encoder_infer_mode": "encoder",
            "prompt_template_video": "li-dit-encode-video",
            "hidden_state_skip_layer": 2,
            "apply_final_norm": True,
            "text_encoder_2": "clipL",
            "text_encoder_precision_2": "fp16",
            "text_states_dim_2": 768,
            "tokenizer_2": "clipL",
            "text_len_2":77,
            "text_projection":"single_refiner",
            }
        args = OmegaConf.create(args_dict)
        if transformer != "none":
            args.ckpt = folder_paths.get_full_path("HunyuanAvatar", transformer)
        else:
            raise Exception("Please download the model first")
        
        # load model
        print("***********Load model ***********")
        hunyuan_video_sampler = tranformer_load(args)
        print("***********Load model done ***********")
        gc_clear()
        return (hunyuan_video_sampler,args,)



class HY_Avatar_EncoderLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "args": ("HY_AVATAR_MODEL_ARGS",),
            },
        }

    RETURN_TYPES = ("MODEL_HY_AVATAR_text_encoder","MODEL_HY_AVATAR_text_encoder_2","HY_AVATAR_MODEL_ARGS")
    RETURN_NAMES = ("text_encoder","text_encoder_2","args")
    FUNCTION = "loader_main"
    CATEGORY = "HunyuanAvatar_Sm"

    def loader_main(self, args,):
        if args.prompt_template_video is not None:
            crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
        else:
            crop_start = 0
        max_length = args.text_len + crop_start
        # prompt_template_video
        prompt_template_video = PROMPT_TEMPLATE[args.prompt_template_video] if args.prompt_template_video is not None else None
        print("="*25, f"load llava", "="*25)
        text_encoder = TextEncoder(text_encoder_type = args.text_encoder,
                                   max_length = max_length,
                                   text_encoder_precision = args.text_encoder_precision,
                                   tokenizer_type = args.tokenizer,
                                   use_attention_mask = args.use_attention_mask,
                                   prompt_template_video = prompt_template_video,
                                   hidden_state_skip_layer = args.hidden_state_skip_layer,
                                   apply_final_norm = args.apply_final_norm,
                                   reproduce = args.reproduce,
                                   logger = logger,
                                   device = 'cpu' if args.cpu_offload else device ,
                                   )
        text_encoder_2 = None
        if args.text_encoder_2 is not None:
            text_encoder_2 = TextEncoder(text_encoder_type=args.text_encoder_2,
                                         max_length=args.text_len_2,
                                         text_encoder_precision=args.text_encoder_precision_2,
                                         tokenizer_type=args.tokenizer_2,
                                         use_attention_mask=args.use_attention_mask,
                                         reproduce=args.reproduce,
                                         logger=logger,
                                         device='cpu' if args.cpu_offload else device , # if not args.use_cpu_offload else 'cpu'
                                         )
        return (text_encoder,text_encoder_2,args)   

class HY_Avatar_PreData:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "image": ("IMAGE",),""
                "text_encoder": ("MODEL_HY_AVATAR_text_encoder",),
                "text_encoder_2": ("MODEL_HY_AVATAR_text_encoder_2",),
                "args": ("HY_AVATAR_MODEL_ARGS",),
                "fps": ("FLOAT", {"default": 25.0, "min": 8.0, "max": 100.0, "step": 1.0}),
                "video_size": ("INT", {"default": 512, "min": 128, "max": 1216, "step": 16}),
                "image_size" : ("INT", {"default": 704, "min": 128, "max": 1216, "step": 16}),
                "video_length": ("INT", {"default": 128, "min": 128, "max": 2048, "step": 4}),
                "prompt":("STRING", {"multiline": True,"default": "A person sits cross-legged by a campfire in a forested area."}),
                "negative_prompt":("STRING", {"multiline": True,"default": "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, Lens changes"}),
                "infer_min":  ("BOOLEAN", {"default": True},),

            }}

    RETURN_TYPES = ("AVATAR_PREDATA",)
    RETURN_NAMES = ("data_dict", )
    FUNCTION = "sampler_main"
    CATEGORY = "HunyuanAvatar_Sm"

    def sampler_main(self, audio, image,text_encoder,text_encoder_2,args,fps,video_size,image_size,video_length,prompt,negative_prompt,infer_min,):
        # save audio to wav file
        audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
        audio_path = os.path.join(folder_paths.get_input_directory(), f"audio_{audio_file_prefix}_temp.wav")
        buff = io.BytesIO()
        torchaudio.save(buff, audio["waveform"].squeeze(0), audio["sample_rate"], format="FLAC")
        with open(audio_path, 'wb') as f:
            f.write(buff.getbuffer())

        wav2vec, feature_extractor, align_instance = audio_image_load(Hunyuan_Avatar_Weigths_Path, device)
        args.video_size=video_size
        if video_length>128:
            infer_min=False
        args.infer_min=infer_min
        args.image_size=image_size
        args.sample_n_frames=video_length+1
   
        kwargs = {
                "text_encoder": text_encoder, 
                "text_encoder_2": text_encoder_2, 
                "feature_extractor": feature_extractor, 
            }
        video_dataset = VideoAudioTextLoaderVal(
                image_size=args.image_size,
                #meta_file=args.input, 
                audio_path=audio_path,
                image_path=tensor_to_pil(image),
                prompt=prompt,
                fps=fps,
                **kwargs,
            )

        sampler = DistributedSampler(video_dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False)
        json_loader = DataLoader(video_dataset, batch_size=1, shuffle=False, sampler=sampler, drop_last=False)
        emb_data=[]

        for index,batch in enumerate(json_loader):
            audio_prompts = batch["audio_prompts"].to(device)
            weight_dtype = audio_prompts.dtype

            audio_prompts = [encode_audio(wav2vec, audio_feat.to(dtype=wav2vec.dtype), fps, num_frames=batch["audio_len"][0]) for audio_feat in audio_prompts]
            audio_prompts = torch.cat(audio_prompts, dim=0).to(device=device, dtype=weight_dtype)
            print(audio_prompts.shape) #torch.Size([1, 272, 10, 5, 384]) #batch["audio_len"] 272
            if audio_prompts.shape[1] <= 129: #补帧足129
                audio_prompts = torch.cat([audio_prompts, torch.zeros_like(audio_prompts[:, :1]).repeat(1,129-audio_prompts.shape[1], 1, 1, 1)], dim=1)
            else:
                audio_prompts = torch.cat([audio_prompts, torch.zeros_like(audio_prompts[:, :1]).repeat(1, 5, 1, 1, 1)], dim=1)
            
            uncond_audio_prompts = torch.zeros_like(audio_prompts[:,:129])
            motion_exp = batch["motion_bucket_id_exps"].to(device)
            motion_pose = batch["motion_bucket_id_heads"].to(device)
            
            pixel_value_ref = batch['pixel_value_ref'].to(device)  # (b f c h w) 取值范围[0,255]
            face_masks = get_facemask(pixel_value_ref.clone(), align_instance, area=3.0) 

            pixel_value_ref = pixel_value_ref.clone().repeat(1,129,1,1,1)
            uncond_pixel_value_ref = torch.zeros_like(pixel_value_ref)
            pixel_value_ref = pixel_value_ref / 127.5 - 1.             
            uncond_pixel_value_ref = uncond_pixel_value_ref * 2 - 1    
            
            pixel_value_ref_for_vae = rearrange(pixel_value_ref, "b f c h w -> b c f h w")
            uncond_uncond_pixel_value_ref = rearrange(uncond_pixel_value_ref, "b f c h w -> b c f h w")

            pixel_value_llava = batch["pixel_value_ref_llava"].to(device)
            pixel_value_llava = rearrange(pixel_value_llava, "b f c h w -> (b f) c h w")
            uncond_pixel_value_llava = pixel_value_llava.clone()
            prompt_embeds, negative_prompt_embeds, prompt_mask, negative_prompt_mask = encode_prompt_audio_text_base(
                    prompt=prompt,
                    uncond_prompt=negative_prompt,
                    pixel_value_llava=pixel_value_llava,
                    uncond_pixel_value_llava=uncond_pixel_value_llava,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,#self.do_classifier_free_guidance,,#TODO
                    negative_prompt=negative_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    lora_scale=None,#TODO
                    clip_skip=None,#TODO
                    text_encoder=text_encoder,
                    data_type="video", 
                    # **kwargs
                )
            prompt_embeds_2, negative_prompt_embeds_2, prompt_mask_2, negative_prompt_mask_2 = encode_prompt_audio_text_base(
                        prompt=prompt,
                        uncond_prompt=negative_prompt,
                        pixel_value_llava=None,
                        uncond_pixel_value_llava=None,
                        device=device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,#self.do_classifier_free_guidance,#TODO
                        negative_prompt=negative_prompt,
                        prompt_embeds=None,
                        negative_prompt_embeds=None,
                        lora_scale=None,#TODO
                        clip_skip=None,#TODO
                        text_encoder=text_encoder_2,
                        # **kwargs
                    )
            batch_dict= {"audio_prompts":audio_prompts,"uncond_audio_prompts":uncond_audio_prompts,"motion_exp":motion_exp,"motion_pose":motion_pose,"face_masks":face_masks,
                         "prompt_embeds":prompt_embeds,"negative_prompt_embeds":negative_prompt_embeds,"prompt_mask":prompt_mask,"negative_prompt_mask":negative_prompt_mask,"prompt_embeds_2":prompt_embeds_2,
                         "negative_prompt_embeds_2":negative_prompt_embeds_2,"prompt_mask_2":prompt_mask_2,"negative_prompt_mask_2":negative_prompt_mask_2,
                         "pixel_value_ref_for_vae":pixel_value_ref_for_vae,"uncond_uncond_pixel_value_ref":uncond_uncond_pixel_value_ref,"pixel_value_llava":pixel_value_llava,"uncond_pixel_value_llava":uncond_pixel_value_llava
                
            }
            emb_data.append(batch_dict)
        wav2vec.to("cpu")
        text_encoder,text_encoder_2=None,None
        gc_clear()
        return ({"json_loader": json_loader, "fps":fps,"emb_data":emb_data,"args":args},)


class HY_Avatar_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_HY_AVATAR_MODEL",),
                "data_dict": ("AVATAR_PREDATA",),  # {}
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 25, "min": 3, "max": 1024, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 7.5, "min": 1, "max": 20, "step": 0.1}),
                 "vae_tiling":  ("BOOLEAN", {"default": True},),
            }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "sampler_main"
    CATEGORY = "HunyuanAvatar_Sm"

    def sampler_main(self, model, data_dict, seed,steps,cfg_scale,vae_tiling):

        print("***********Start infer  ***********")
        args=data_dict.get("args")
        args.seed = seed
        args.infer_steps = steps
        args.cfg_scale = cfg_scale
        args.vae_tiling = vae_tiling
        iamge = hunyuan_avatar_main(args,model,data_dict.get("json_loader"),data_dict.get("emb_data"),args.infer_min)
        gc.collect()
        torch.cuda.empty_cache()
        return (load_images(iamge), data_dict.get("fps"))


NODE_CLASS_MAPPINGS = {
    "HY_Avatar_Loader": HY_Avatar_Loader,
    "HY_Avatar_EncoderLoader": HY_Avatar_EncoderLoader,
    "HY_Avatar_PreData": HY_Avatar_PreData,
    "HY_Avatar_Sampler": HY_Avatar_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HY_Avatar_Loader": "HY_Avatar_Loader",
    "HY_Avatar_EncoderLoader": "HY_Avatar_EncoderLoader",
    "HY_Avatar_PreData": "HY_Avatar_PreData",
    "HY_Avatar_Sampler": "HY_Avatar_Sampler",
}
