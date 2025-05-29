import os
import numpy as np
import torch
import warnings
import threading
import traceback
import uvicorn
from fastapi import FastAPI, Body
from pathlib import Path
from datetime import datetime
import torch.distributed as dist
from hymm_gradio.tool_for_end2end import *
from hymm_sp.config import parse_args
from hymm_sp.sample_inference_audio import HunyuanVideoSampler

from hymm_sp.modules.parallel_states import (
    initialize_distributed,
    nccl_info,
)

from transformers import WhisperModel
from transformers import AutoFeatureExtractor
from hymm_sp.data_kits.face_align import AlignImage


warnings.filterwarnings("ignore")
MODEL_OUTPUT_PATH = os.environ.get('MODEL_BASE')
app = FastAPI()
rlock = threading.RLock()



@app.api_route('/predict2', methods=['GET', 'POST'])
def predict(data=Body(...)):
    is_acquire = False
    error_info = ""
    try:
        is_acquire = rlock.acquire(blocking=False)
        if is_acquire:
            res = predict_wrap(data)
            return res
    except Exception as e:
        error_info = traceback.format_exc()
        print(error_info)
    finally:
        if is_acquire:
            rlock.release()
    return {"errCode": -1, "info": "broken"}

def predict_wrap(input_dict={}):
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        rank = local_rank = torch.distributed.get_rank()
        print(f"sp_size={nccl_info.sp_size}, rank {rank} local_rank {local_rank}")
    try:
        print(f"----- rank = {rank}")
        if rank == 0:
            input_dict = process_input_dict(input_dict)

            print('------- start to predict -------')
            # Parse input arguments
            image_path = input_dict["image_path"]
            driving_audio_path = input_dict["audio_path"]

            prompt = input_dict["prompt"]

            save_fps = input_dict.get("save_fps", 25)


            ret_dict = None
            if image_path is None or driving_audio_path is None:
                ret_dict = {
                    "errCode": -3, 
                    "content": [
                        {
                            "buffer": None
                        },
                    ], 
                    "info": "input content is not valid", 
                }

                print(f"errCode: -3, input content is not valid!")
                return ret_dict

            # Preprocess input batch
            torch.cuda.synchronize()

            a = datetime.now()
            
            try:
                model_kwargs_tmp = data_preprocess_server(
                                        args, image_path, driving_audio_path, prompt, feature_extractor
                                        )
            except:
                ret_dict = {
                    "errCode": -2,         
                    "content": [
                            {
                                "buffer": None
                            },
                        ],
                    "info": "failed to preprocess input data"
                }
                print(f"errCode: -2, preprocess failed!")
                return ret_dict

            text_prompt = model_kwargs_tmp["text_prompt"]
            audio_path = model_kwargs_tmp["audio_path"]
            image_path = model_kwargs_tmp["image_path"]
            fps = model_kwargs_tmp["fps"]
            audio_prompts = model_kwargs_tmp["audio_prompts"]
            audio_len = model_kwargs_tmp["audio_len"]
            motion_bucket_id_exps = model_kwargs_tmp["motion_bucket_id_exps"]
            motion_bucket_id_heads = model_kwargs_tmp["motion_bucket_id_heads"]
            pixel_value_ref = model_kwargs_tmp["pixel_value_ref"]
            pixel_value_ref_llava = model_kwargs_tmp["pixel_value_ref_llava"]
            


            torch.cuda.synchronize()
            b = datetime.now()
            preprocess_time = (b - a).total_seconds()
            print("="*100)
            print("preprocess time :", preprocess_time)
            print("="*100)
            
        else:
            text_prompt = None
            audio_path = None
            image_path = None
            fps = None
            audio_prompts = None
            audio_len = None
            motion_bucket_id_exps = None
            motion_bucket_id_heads = None
            pixel_value_ref = None
            pixel_value_ref_llava = None

    except:
        traceback.print_exc()
        if rank == 0:
            ret_dict = {
                "errCode": -1,         # Failed to generate video
                "content":[
                    {
                        "buffer": None
                    }
                ],
                "info": "failed to preprocess",
            }
            return ret_dict

    try:
        broadcast_params = [
            text_prompt,
            audio_path,
            image_path,
            fps,
            audio_prompts,
            audio_len,
            motion_bucket_id_exps,
            motion_bucket_id_heads,
            pixel_value_ref,
            pixel_value_ref_llava,
        ]
        dist.broadcast_object_list(broadcast_params, src=0)
        outputs = generate_image_parallel(*broadcast_params)

        if rank == 0:
            samples = outputs["samples"]
            sample = samples[0].unsqueeze(0)

            sample = sample[:, :, :audio_len[0]]
            
            video = sample[0].permute(1, 2, 3, 0).clamp(0, 1).numpy()
            video = (video * 255.).astype(np.uint8)

            output_dict = {
                "err_code": 0, 
                "err_msg": "succeed", 
                "video": video, 
                "audio": input_dict.get("audio_path", None), 
                "save_fps": save_fps, 
            }

            ret_dict = process_output_dict(output_dict)
            return ret_dict
    
    except:
        traceback.print_exc()
        if rank == 0:
            ret_dict = {
                "errCode": -1,         # Failed to generate video
                "content":[
                    {
                        "buffer": None
                    }
                ],
                "info": "failed to generate video",
            }
            return ret_dict
        
    return None
    
def generate_image_parallel(text_prompt,
                    audio_path,
                    image_path,
                    fps,
                    audio_prompts,
                    audio_len,
                    motion_bucket_id_exps,
                    motion_bucket_id_heads,
                    pixel_value_ref,
                    pixel_value_ref_llava
                    ):
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")

    batch = {
        "text_prompt": text_prompt,
        "audio_path": audio_path,
        "image_path": image_path,
        "fps": fps,
        "audio_prompts": audio_prompts,
        "audio_len": audio_len,
        "motion_bucket_id_exps": motion_bucket_id_exps,
        "motion_bucket_id_heads": motion_bucket_id_heads,
        "pixel_value_ref": pixel_value_ref,
        "pixel_value_ref_llava": pixel_value_ref_llava
    }

    samples = hunyuan_sampler.predict(args, batch, wav2vec, feature_extractor, align_instance)
    return samples

def worker_loop():
    while True:
        predict_wrap()
        

if __name__ == "__main__":
    audio_args = parse_args()
    initialize_distributed(audio_args.seed)
    hunyuan_sampler = HunyuanVideoSampler.from_pretrained(
        audio_args.ckpt, args=audio_args)
    args = hunyuan_sampler.args
    
    rank = local_rank = 0
    device = torch.device("cuda")
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        rank = local_rank = torch.distributed.get_rank()

    feature_extractor = AutoFeatureExtractor.from_pretrained(f"{MODEL_OUTPUT_PATH}/ckpts/whisper-tiny/")
    wav2vec = WhisperModel.from_pretrained(f"{MODEL_OUTPUT_PATH}/ckpts/whisper-tiny/").to(device=device, dtype=torch.float32)
    wav2vec.requires_grad_(False)


    BASE_DIR = f'{MODEL_OUTPUT_PATH}/ckpts/det_align/'
    det_path = os.path.join(BASE_DIR, 'detface.pt')    
    align_instance = AlignImage("cuda", det_path=det_path)



    if rank == 0:
        uvicorn.run(app, host="0.0.0.0", port=80)
    else:
        worker_loop()
    
