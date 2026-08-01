from datasets import load_dataset, concatenate_datasets
import av
import os
import numpy as np
import torch
from ..model.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

def read_video_pyav(container, indices=None):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)

    if indices is None:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        print(f"INFO: {len(frames)} frames are decoded.")
        return np.stack(frames)
    else:
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)   
        print(f"INFO: {len(frames)} frames are decoded.")
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
def clip_input_video( base_model_path,task, data_instance, frame_num=64, model_type='llava_ov',data_path=None):
    from transformers import AutoProcessor

    if "Qwen2.5-VL" in base_model_path:
        processor = Qwen2_5_VLProcessor.from_pretrained(
            base_model_path,  device_map="auto" ,torch_dtype="auto",
        )
    else:
        processor = AutoProcessor.from_pretrained(base_model_path)

    if model_type == 'llava_ov':
        if task == "VideoDetailCaption":
            video_path = os.path.join(data_path, "Test_Videos/")
            video_name = data_instance["video_name"]
            video_path = video_path + video_name + ".mp4"

            question = data_instance["question"]
            conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": question},
                    ],
            },
            ]
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            # print("Total frames:",total_frames)
            indices = np.arange(0, total_frames, total_frames / frame_num).astype(int)
            video = read_video_pyav(container, indices)
            
        elif task == "MVBench":
            video_path = data_path
            video_name = data_instance["video"]
            video_path = video_path + video_name

            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": question},
                    ],
            },
            ]

            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            # print("Total frames:",total_frames)
            indices = np.arange(0, total_frames, total_frames / frame_num).astype(int)
            video = read_video_pyav(container, indices)

        elif task == 'LongVideoBench':
            video_path = data_path
            video_name = data_instance["video_path"]
            video_path = video_path + video_name

            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": question},
                    ],
            },
            ]

            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            # print("Total frames:",total_frames)
            if total_frames == 0:
                return None
            indices = np.arange(0, total_frames, total_frames / frame_num).astype(int)
            video = read_video_pyav(container, indices)

        elif task == "MVLU":
            video_reader = data_instance['video']

            total_frames = len(video_reader)
            # print("Total frames:", total_frames)
            indices = np.linspace(0, total_frames - 1, frame_num, dtype=int)
            video = video_reader.get_batch(indices).asnumpy()

            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": question},
                    ],
            },
            ]

        # display_frame_grid(video)
        # save_frames(video)
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(videos=list(video), text=prompt, return_tensors="pt").to("cuda")

    elif model_type == 'qwen2_5_vl':
        def calculate_fps_for_target_frames(container, target_frames):
            video_stream = container.streams.video[0]
            duration = container.duration / 1000000
            if duration <= 0:
                return 1.0 
            
            required_fps = target_frames / duration
            # print(f"INFO: Duration: {duration:.2f}s, frame_num: {target_frames}, fps: {required_fps:.2f}")
            return required_fps

        if task == "VideoDetailCaption":
            video_path = os.path.join(data_path, "Test_Videos/")
            video_name = data_instance["video_name"]
            video_path = video_path + video_name + ".mp4"
            question = data_instance["question"]
            # print("video_name",video_name)
            # print("video_path",video_path)
            # print("question",question)
            # print("answer",data_instance["answer"])
        
        elif task == "MVBench":
            video_path = data_path
            video_name = data_instance["video"]
            video_path = video_path + video_name
            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            
        elif task == 'LongVideoBench':
            video_path = data_path
            video_name = data_instance["video_path"]
            video_path = video_path + video_name
            question = "Please provide a detailed description of the video, focusing on the main subjects, their actions, and the background scenes."
            
        # elif task == "MVLU":
        #     video_reader = data_instance['video']
        #     total_frames = len(video_reader)
        #     print("Total frames:", total_frames)

        #     indices = np.linspace(0, total_frames - 1, frame_num, dtype=int)
        #     frames = video_reader.get_batch(indices).asnumpy()
        
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        # print("Total frames:", total_frames)
        print("video_path",video_path)
        if total_frames == 0:
            return None
        print("video_path",video_path)

        fps = calculate_fps_for_target_frames(container, frame_num)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"file://{video_path}",
                        "max_pixels": 448*448,  
                        "fps": fps, 
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        # print("messages",messages)
        
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # print("text",text)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
       
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")
    
    # print("INFO: Input length:", inputs['input_ids'].shape[1])
    return inputs