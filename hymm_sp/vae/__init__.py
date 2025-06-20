import torch
import os
from pathlib import Path
from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ..constants import VAE_PATH, PRECISION_TO_TYPE

def load_vae(vae_type,
             vae_precision=None,
             sample_size=None,
             vae_path=None,
             logger=None,
             device=None,
             cpu_offload=False,
             
             ):
    if vae_path is None:
        vae_path = VAE_PATH[vae_type]
    vae_compress_spec, _, _ = vae_type.split("-")
    length = len(vae_compress_spec)
    if length == 3:
        # if logger is not None:
        #     logger.info(f"Loading 3D VAE model ({vae_type}) from: {vae_path}")

        import json
        with open(os.path.join(vae_path, "config.json"), "r", encoding="utf-8") as reader:
            text = reader.read()
        vae_config= json.loads(text)
        # reduce time window used by the VAE for temporal splitting (former time windows is too large for 24 GB) 
        if vae_config["sample_tsize"] == 64:
            vae_config["sample_tsize"] = 32 
        with open(os.path.join(vae_path, "config.json"), "w", encoding="utf-8") as writer:
            writer.write(json.dumps(vae_config))


        config = AutoencoderKLCausal3D.load_config(vae_path)
        if sample_size:
            vae = AutoencoderKLCausal3D.from_config(config, sample_size=sample_size)
        else:
            vae = AutoencoderKLCausal3D.from_config(config)
        ckpt = torch.load(Path(vae_path) / "pytorch_model.pt", map_location=vae.device)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        # vae_ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
        vae_ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items()}  
        vae.load_state_dict(vae_ckpt)
       
        spatial_compression_ratio = vae.config.spatial_compression_ratio
        time_compression_ratio = vae.config.time_compression_ratio
    else:
        raise ValueError(f"Invalid VAE model: {vae_type}. Must be 3D VAE in the format of '???-*'.")

    if vae_precision is not None:
        vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    vae.requires_grad_(False)

    if logger is not None:
        logger.info(f"VAE to dtype: {vae.dtype}")

    if device is not None:
        vae = vae.to(device)

    # Set vae to eval mode, even though it's dropout rate is 0.
    vae.eval()
    if cpu_offload:
        vae.is_cpu_offload()
    return vae, vae_path, spatial_compression_ratio, time_compression_ratio
