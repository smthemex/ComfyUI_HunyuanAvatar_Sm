# Download Pretrained Models

All models are stored in `HunyuanVideo-Avatar/weights` by default, and the file structure is as follows
```shell
HunyuanVideo-Avatar
  ├──weights
  │  ├──ckpts
  │  │  ├──README.md
  │  │  ├──hunyuan-video-t2v-720p
  │  │  │  ├──transformers
  │  │  │  │  ├──mp_rank_00_model_states.pt
  │  │  │  │  ├──mp_rank_00_model_states_fp8.pt
  │  │  │  │  ├──mp_rank_00_model_states_fp8_map.pt
  │  │  │  ├──vae
  │  │  │  │  ├──pytorch_model.pt
  │  │  │  │  ├──config.json
  │  │  ├──llava_llama_image
  │  │  │  ├──model-00001-of-00004.safatensors
  │  │  │  ├──model-00002-of-00004.safatensors
  │  │  │  ├──model-00003-of-00004.safatensors
  │  │  │  ├──model-00004-of-00004.safatensors
  │  │  │  ├──...
  │  │  ├──text_encoder_2
  │  │  ├──whisper-tiny
  │  │  ├──det_align
  │  │  ├──...
```

## Download HunyuanVideo-Avatar model
To download the HunyuanCustom model, first install the huggingface-cli. (Detailed instructions are available [here](https://huggingface.co/docs/huggingface_hub/guides/cli).)

```shell
python -m pip install "huggingface_hub[cli]"
```

Then download the model using the following commands:

```shell
# Switch to the directory named 'HunyuanVideo-Avatar/weights'
cd HunyuanVideo-Avatar/weights
# Use the huggingface-cli tool to download HunyuanVideo-Avatar model in HunyuanVideo-Avatar/weights dir.
# The download time may vary from 10 minutes to 1 hour depending on network conditions.
huggingface-cli download tencent/HunyuanVideo-Avatar --local-dir ./
```
