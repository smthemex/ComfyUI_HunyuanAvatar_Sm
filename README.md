# ComfyUI_HunyuanAvatar_Sm
* [HunyuanVideo-Avatar](https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar): High-Fidelity Audio-Driven Human Animation for Multiple Characters,try it in comfyUI ,if your VRAM >24G

TIPS:
-----
* 因为测试的是cpu卸载方案，所以很大几率会报错.目前主要处理数据混杂（音频、图片、text的emb，多种模型）因为cpu卸载带来的各种不兼容，计划能跑才上云。
* 测试环境:VRam12G，Ram64G,transformer必须指定版本，否则会报错，腾讯都没法解决，所以老实匹配。


1.Installation  
-----
In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_HunyuanAvatar_Sm.git
```  
  
2.requirements  
----
```
pip install -r requirements.txt
```

3 models 
----
* download files from [tencent/HunyuanVideo-Avatar](https://huggingface.co/tencent/HunyuanVideo-Avatar) 
```
├── ComfyUI/models/HunyuanAvatar/
|   ├── det_align/
|         ├──detface.pt
|   ├── llava_llama_image/
|         ├──config.json
|         ├── ...所有json文件以及所有safetensors模型
|   ├──text_encoder_2/
|         ├──config.json
|         ├── ... 所有json文件以及model.safetensors模型
|   ├──vae/
|         ├──config.json
|         ├── pytorch_model.pt
|   ├──whisper-tiny/
|         ├──config.json
|         ├── ... 所有json文件以及model.safetensors模型
|   ├── mp_rank_00_model_states_fp8_map.pt #104K
|   ├── mp_rank_00_model_states_fp8.pt.pt #24.9G
```
4 example
----
![](https://github.com/smthemex/ComfyUI_HunyuanAvatar_Sm/blob/main/example_workflows/example.png)


## 🔗 BibTeX

If you find [HunyuanVideo-Avatar](https://arxiv.org/pdf/2505.20156) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{hu2025HunyuanVideo-Avatar,
      title={HunyuanVideo-Avatar: High-Fidelity Audio-Driven Human Animation for Multiple Characters}, 
      author={Yi Chen and Sen Liang and Zixiang Zhou and Ziyao Huang and Yifeng Ma and Junshu Tang and Qin Lin and Yuan Zhou and Qinglin Lu},
      year={2025},
      eprint={2505.20156},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/pdf/2505.20156}, 
}
```

## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration. 
