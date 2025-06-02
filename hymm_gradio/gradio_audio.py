import os
import cv2
import glob
import json
import datetime
import requests
import gradio as gr
from tool_for_end2end import *

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
DATADIR = './temp'
_HEADER_ = '''
<div style="text-align: center; max-width: 650px; margin: 0 auto;">
    <h1 style="font-size: 2.5rem; font-weight: 700; margin-bottom: 1rem; display: contents;">Tencent HunyuanVideo-Avatar Demo</h1>
</div>

''' 
# flask url
URL = "http://127.0.0.1:80/predict2"

def post_and_get(audio_input, id_image, prompt):
    now = datetime.datetime.now().isoformat()
    imgdir = os.path.join(DATADIR, 'reference')
    videodir = os.path.join(DATADIR, 'video')
    imgfile = os.path.join(imgdir, now + '.png')
    output_video_path = os.path.join(videodir, now + '.mp4')


    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(videodir, exist_ok=True)
    cv2.imwrite(imgfile, id_image[:,:,::-1])

    proxies = {
        "http": None, 
        "https": None,
    }
    
    files = {
        "image_buffer": encode_image_to_base64(imgfile), 
        "audio_buffer":  encode_wav_to_base64(audio_input),
        "text": prompt,
        "save_fps": 25, 
    }
    r = requests.get(URL, data = json.dumps(files), proxies=proxies)
    ret_dict = json.loads(r.text)
    print(ret_dict["info"])
    save_video_base64_to_local(
        video_path=None, 
        base64_buffer=ret_dict["content"][0]["buffer"], 
        output_video_path=output_video_path)


    return output_video_path

def create_demo():
    
    with gr.Blocks() as demo:
        gr.Markdown(_HEADER_)
        with gr.Tab('语音数字人驱动'):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        prompt = gr.Textbox(label="Prompt", value="a man is speaking.")
  
                    audio_input = gr.Audio(sources=["upload"],
                                       type="filepath",
                                       label="Upload Audio",
                                       elem_classes="media-upload",
                                       scale=1)
                    id_image = gr.Image(label="Input reference image", height=480)

                with gr.Column(scale=2):
                    with gr.Group():
                        output_image = gr.Video(label="Generated Video")
                        

                with gr.Column(scale=1):
                    generate_btn = gr.Button("Generate")

            generate_btn.click(fn=post_and_get,
                inputs=[audio_input, id_image, prompt],
                outputs=[output_image],
            )
            
            # quick_prompts = [[x] for x in glob.glob('./assets/images/*.png')]
            # example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Other object', samples_per_page=1000, components=[id_image])
            # example_quick_prompts.click(lambda x: x[0], inputs=example_quick_prompts, outputs=id_image, show_progress=False, queue=False)
            # with gr.Row(), gr.Column():
            #     gr.Markdown("## Examples")
            #     example_inps = [
            #         [
            #             'A woman is drinking coffee at a café.',
            #             './assets/images/seg_woman_01.png',
            #             1280, 720, 30, 129, 7.5, 13, 1024,
            #             "assets/videos/seg_woman_01.mp4"
            #         ],
            #         [
            #             'In a cubicle of an office building, a woman focuses intently on the computer screen, typing rapidly on the keyboard, surrounded by piles of documents.',
            #             './assets/images/seg_woman_03.png',
            #             1280, 720, 30, 129, 7.5, 13, 1025,
            #             "./assets/videos/seg_woman_03.mp4"
            #         ],
            #         [
            #             'A man walks across an ancient stone bridge holding an umbrella, raindrops tapping against it.',
            #             './assets/images/seg_man_01.png',
            #             1280, 720, 30, 129, 7.5, 13, 1025,
            #             "./assets/videos/seg_man_01.mp4"
            #         ],
            #         [
            #             'During a train journey, a man admires the changing scenery through the window.',
            #             './assets/images/seg_man_02.png',
            #             1280, 720, 30, 129, 7.5, 13, 1026,
            #             "./assets/videos/seg_man_02.mp4"
            #         ]
            #     ]
            #     gr.Examples(examples=example_inps, inputs=[prompt, id_image, width, height, num_steps, num_frames, guidance, flow_shift, seed, output_image],)
    return demo

if __name__ == "__main__":
    allowed_paths = ['/']
    demo = create_demo()
    demo.launch(server_name='0.0.0.0', server_port=8080, share=True, allowed_paths=allowed_paths)
