# ComfyUI_HunyuanAvatar_Sm
* [HunyuanVideo-Avatar](https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar): High-Fidelity Audio-Driven Human Animation for Multiple Characters,try it in comfyUI ,if your VRAM >12 or 10G

TIPS:
-----
* 黑图的原因可能是是diffuser版本限定33.0，以及precision我强行改成fb16的缘故，终于复现了。测试环境，cu124，torch2.6  4070


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
|   ├── mp_rank_00_model_states_fp8_map.pt #104K if use fp8  如果用fp8则下载
|   ├── mp_rank_00_model_states_fp8.pt.pt #24.9G  if use fp8  如果用fp8则下载
|   ├──mp_rank_00_model_states.pt
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
