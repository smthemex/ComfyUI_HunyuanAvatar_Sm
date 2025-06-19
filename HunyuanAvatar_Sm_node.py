# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import random
import io
import torchaudio
from einops import rearrange
from omegaconf import OmegaConf
from .node_utils import tensor2pil_upscale,gc_clear,trim_audio
from .hymm_sp.sample_gpu_poor import tranformer_load,audio_image_load
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
            "linear_schedule_end": 25,
            "flow_solver": "euler",
            "flow_reverse": True,
            "save_path": folder_paths.get_output_directory(),
            "use_fp8": use_fp8,
            "cpu_offload": cpu_offload,
            "infer_min": True,
            "precision": "bf16",#bf16
            "reproduce": True,
            "num_images": 1,
            "val_disable_autocast": False,
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
            "apply_final_norm": True, # NEED CHECK
            "text_encoder_2": "clipL",
            "text_encoder_precision_2": "fp16",
            "text_states_dim_2": 768,
            "tokenizer_2": "clipL",
            "text_len_2":77,
            "text_projection":"single_refiner",
            "daul_role":False,
            "face_size": 3.0,
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



class HY_Avatar_PreData:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_HY_AVATAR_MODEL",),
                "args": ("HY_AVATAR_MODEL_ARGS",),
                "audio": ("AUDIO",),
                "image": ("IMAGE",),""
                "fps": ([25.0, 12.5],),
                "width": ("INT", {"default": 512, "min": 128, "max": 1216, "step": 64}),
                "height": ("INT", {"default": 512, "min": 128, "max": 1216, "step": 64}),
                "face_size": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "image_size" : ("INT", {"default": 704, "min": 128, "max": 1216, "step": 64}),
                "video_length": ("INT", {"default": 128, "min": 128, "max": 2048, "step": 4}),
                "prompt":("STRING", {"multiline": True,"default": "A person sits cross-legged by a campfire in a forested area."}),
                "negative_prompt":("STRING", {"multiline": True,"default": "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, Lens changes"}),
                "duration": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 100000000000.0, "step": 0.1}),
                "infer_min":  ("BOOLEAN", {"default": True},),
                "object_name": ("STRING", {"multiline": False,"default": "girl"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 25, "min": 3, "max": 1024, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 7.5, "min": 1, "max": 20, "step": 0.1}),
                "vae_tiling":  ("BOOLEAN", {"default": True},),},
            "optional":{
                "audio_d": ("AUDIO",),
                            }

            }

    RETURN_TYPES = ("MODEL_HY_AVATAR_MODEL","AVATAR_PREDATA","HY_AUDIO_MODEL","AUDIO")
    RETURN_NAMES = ("model", "json_loader", "audio_model","audio")
    FUNCTION = "sampler_main"
    CATEGORY = "HunyuanAvatar_Sm"

    def sampler_main(self, model,args,audio, image,fps,width,height,face_size,image_size,video_length,prompt,negative_prompt,duration,infer_min,object_name,seed,steps,cfg_scale,vae_tiling,**kwargs):
        #pre audio files
        audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
        audio_path = os.path.join(folder_paths.get_input_directory(), f"audio_{audio_file_prefix}_temp.wav")
        if isinstance(kwargs.get("audio_d"),dict):
            audio2 = kwargs.get("audio_d")
            daul_role=True
            if audio["sample_rate"] != audio2["sample_rate"]:
                raise ValueError("two audios must has same sample_rate 采样率不一致，无法直接拼接")
            actual_duration1 = min(duration, audio["waveform"].shape[1] / audio["sample_rate"])
            actual_duration2 = min(duration, audio2["waveform"].shape[1] / audio2["sample_rate"])
            min_duration = min(actual_duration1, actual_duration2)
            trimmed1 = trim_audio(audio, min_duration)   #trim audio
            trimmed2 = trim_audio(audio2, min_duration) 
            waveform = torch.cat([trimmed1, trimmed2], dim=1) 
            infer_duration=min_duration*2
        else:
            waveform=audio["waveform"].squeeze(0)
            daul_role=False
            num_frames = waveform.shape[1]
            duration_input = num_frames / audio["sample_rate"]
            infer_duration=min(duration,duration_input)

            
        print(f" infer audio duration is: {duration} seconds.use {infer_duration} seconds to infer.")   

        # save audio to wav file
        buff = io.BytesIO()
        torchaudio.save(buff, waveform, audio["sample_rate"], format="FLAC")
        with open(audio_path, 'wb') as f:
            f.write(buff.getbuffer())

        # load audio model
        wav2vec, feature_extractor, align_instance = audio_image_load(Hunyuan_Avatar_Weigths_Path, device)

        # args
        args.daul_role=daul_role
        args.face_size=face_size
        if video_length>128:
            infer_min=False
        args.infer_min=infer_min
        args.image_size=image_size
        args.sample_n_frames=video_length+1
        args.pos_prompt=prompt
        args.neg_prompt=negative_prompt
        args.cfg_scale=cfg_scale
        args.steps=steps
        args.seed=seed
        args.vae_tiling=vae_tiling
        
        # pre data
        kwargs = {
                "text_encoder": model.text_encoder, 
                "text_encoder_2": model.text_encoder_2, 
                "feature_extractor": feature_extractor, 
            }
        video_dataset = VideoAudioTextLoaderVal(
                image_size=args.image_size,
                #meta_file=args.input, 
                audio_path=audio_path,
                image_path=tensor2pil_upscale(image,width,height),
                prompt=prompt,
                fps=fps,
                infer_duration=infer_duration,
                name=object_name,
                **kwargs,
            )

        #sampler = DistributedSampler(video_dataset, num_replicas=1, rank=0, shuffle=False, drop_last=False)
        json_loader = DataLoader(video_dataset, batch_size=1, shuffle=False,drop_last=False)
        if daul_role:
            audio= {"waveform": waveform.unsqueeze(0), "sample_rate": audio["sample_rate"]} #need check
        gc_clear()
        return (model,json_loader,{"wav2vec": wav2vec, "feature_extractor": feature_extractor, "align_instance":align_instance, "fps":fps}, audio) 


class HY_Avatar_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_HY_AVATAR_MODEL",),
                "json_loader": ("AVATAR_PREDATA",),  # {}
                "audio_model": ("HY_AUDIO_MODEL",),
              
            }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "sampler_main"
    CATEGORY = "HunyuanAvatar_Sm"

    def sampler_main(self, model, json_loader,audio_model,):
        videolist = []
        for batch_index, batch in enumerate(json_loader, start=1):

            if model.args.infer_min:
                batch["audio_len"][0] = 129
            # fps = batch["fps"]    
            # audio_path = batch["audio_path"][0]
            samples = model.predict(model.args, batch, audio_model["wav2vec"], audio_model["feature_extractor"], audio_model["align_instance"])
            
            sample = samples['samples'][0].unsqueeze(0)                    # denoised latent, (bs, 16, t//4, h//8, w//8)
            sample = sample[:, :, :batch["audio_len"][0]]
            
            video = rearrange(sample[0], "c f h w -> f h w c")
            videolist.append(video)
            # video_ = (video * 255.).data.cpu().numpy().astype(np.uint8)  # （f h w c)
            # torch.cuda.empty_cache()

            # final_frames = []
            # for frame in video_:
            #     final_frames.append(frame)
            # final_frames = np.stack(final_frames, axis=0)
            # import imageio
            # output_path=os.path.join(folder_paths.get_output_directory(), f"video_{batch_index}.mp4")
           
            # imageio.mimsave(output_path, final_frames, fps=fps.item())
            # #os.system(f"ffmpeg -i '{output_path}' -i '{audio_path}' -shortest '{output_audio_path}' -y -loglevel quiet; rm '{output_path}'")

            gc_clear()


        return (videolist[0], audio_model["fps"])


NODE_CLASS_MAPPINGS = {
    "HY_Avatar_Loader": HY_Avatar_Loader,
    "HY_Avatar_PreData": HY_Avatar_PreData,
    "HY_Avatar_Sampler": HY_Avatar_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HY_Avatar_Loader": "HY_Avatar_Loader",
    "HY_Avatar_PreData": "HY_Avatar_PreData",
    "HY_Avatar_Sampler": "HY_Avatar_Sampler",
}
