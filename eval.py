import argparse
import os
import logging
from omegaconf import OmegaConf

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from transformers import CLIPTextModel, CLIPTokenizer

from contextualstory.models.videoldm_unet import VideoLDMUNet3DConditionModel
from contextualstory.pipelines.pipeline_conditional_animation import ConditionalAnimationPipeline
from contextualstory.utils.util import save_videos_grid
from diffusers.utils.import_utils import is_xformers_available
from contextualstory.models.text_adapter import TextAdapter
from contextualstory.data.pororo import PororoDataset
from contextualstory.data.flintstones_data import FlintstonesDataset

import torch
import os
import numpy as np
from torchvision import transforms
import random
from PIL import Image, ImageDraw, ImageFont
from accelerate import PartialState
from accelerate import Accelerator
from itertools import chain
import textwrap
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from typing import Dict, Optional, Tuple
from pathlib import Path

import json

diffusers.utils.logging.disable_progress_bar()


def plot_generated_images(images, new_images, texts, video_length=5):
    num_image = len(images)
    resized_width, resized_height = 128, 128
    text_box_height = 200
    font_path = "./utils/Times_New_Roman.ttf" 
    font_size = 12

    image_canvas = Image.new('RGB', (num_image * resized_width, resized_height))
    new_image_canvas = Image.new('RGB', (num_image * resized_width, resized_height))

    for i in range(num_image):
        resized_image = images[i].resize((resized_width, resized_height))
        image_canvas.paste(resized_image, (i * resized_width, 0))

        resized_new_image = new_images[i].resize((resized_width, resized_height))
        new_image_canvas.paste(resized_new_image, (i * resized_width, 0))

    combined_image = Image.new('RGB', (video_length * resized_width, 2 * resized_height + 200), color='white')
    combined_image.paste(image_canvas, (0, 200))
    combined_image.paste(new_image_canvas, (0, 200+resized_height))


    # 加载 Times New Roman 字体
    font = ImageFont.truetype(font_path, font_size)

    text_content = texts

    text_box = Image.new('RGB', (video_length * resized_width, text_box_height), color='white')
    text_draw = ImageDraw.Draw(text_box)

    # 将文字逐行添加到文字框，处理自动换行
    y_offset = 0
    for i, line in enumerate(text_content):
        wrapped_lines = textwrap.wrap(line, width=resized_width-5)  # 设置每行最大宽度
        for j, wrapped_line in enumerate(wrapped_lines):
            if i==0 and j==0:
                y_offset += 5
            else:
                y_offset += font.getsize(wrapped_lines[0])[1] + 5  # 设置行间距
            text_draw.text((5, y_offset), wrapped_line, fill='black', font=font)

    # 将文字框粘贴到大图的顶部
    combined_image.paste(text_box, (0, 0))
    return combined_image


def save_generated_images(images, new_images, texts, img_ids, output_dir, initial_idx, video_length=5):
    gen_img_save_dir = os.path.join(output_dir, "gen_img")
    ori_img_save_dir = os.path.join(output_dir, "ori_img")
    texts_save_dir = os.path.join(output_dir, "texts")
    figures_save_dir = os.path.join(output_dir, "figures") 

    if not os.path.exists(gen_img_save_dir):
        os.makedirs(gen_img_save_dir)
    if not os.path.exists(ori_img_save_dir):
        os.makedirs(ori_img_save_dir)
    if not os.path.exists(texts_save_dir):
        os.makedirs(texts_save_dir)
    if not os.path.exists(figures_save_dir):
        os.makedirs(figures_save_dir)

    for idx, i in enumerate(range(0, len(images), video_length)):
        sub_images = images[i:i+video_length]
        sub_new_images = new_images[i:i+video_length]
        sub_texts = texts[i:i+video_length]
        sub_img_ids = img_ids[i:i+video_length]
        figures = plot_generated_images(sub_images, sub_new_images, sub_texts, video_length)

        image_idx = initial_idx+idx

        for i, img in enumerate(sub_images):
            img.save(os.path.join(ori_img_save_dir, f"ori_img_{image_idx:04d}_{i}.png"))
        for i, img in enumerate(sub_new_images):
            img.save(os.path.join(gen_img_save_dir, f"{sub_img_ids[i]}.png"))

        with open(os.path.join(texts_save_dir, f'text_{image_idx:04d}.txt'), 'w') as f:
            for text in sub_texts:
                f.write(text+'\n')

        figures.save(os.path.join(figures_save_dir, f"figure_{image_idx:04d}.png"))
        print(f"Save result {image_idx:04d}")
   


def main(config):

    accelerator = Accelerator()
    pipeline_path = os.path.join(config.output_dir, config.name, "final_checkpoint")
    test_result_dir = os.path.join(config.output_dir, config.name, "result")

    if config.resume_from_checkpoint:
        print(config.resume_from_checkpoint)
        test_result_dir=os.path.join(config.resume_from_checkpoint, "result")
        unet = VideoLDMUNet3DConditionModel.from_pretrained(config.resume_from_checkpoint, subfolder="unet", use_safetensors=True, torch_dtype=torch.float16)
        pipeline = ConditionalAnimationPipeline.from_pretrained(pipeline_path, unet=unet, torch_dtype=torch.float16)
    else:
        pipeline = ConditionalAnimationPipeline.from_pretrained(pipeline_path, torch_dtype=torch.float16)
    distributed_state = PartialState()
    pipeline.to(distributed_state.device)
    pipeline.set_progress_bar_config(disable=True)


    if config.frameinit_kwargs.enable:
        pipeline.init_filter(
            width         = config.sampling_kwargs.width,
            height        = config.sampling_kwargs.height,
            video_length  = config.sampling_kwargs.n_frames,
            filter_params = config.frameinit_kwargs.filter_params,
        )
    
    motion_cond=None
    if config.unet_additional_kwargs.use_motion_condition:
        if False: #config.resume_from_checkpoint:
            with open(os.path.join(config.resume_from_checkpoint ,"cumulative_motion.json"), 'r') as f:
                cumulative_motion=json.load(f)
                motion_cond = torch.tensor(cumulative_motion, dtype=torch.float16, device=distributed_state.device) # [1, 31]
        else:
            with open(os.path.join(config.output_dir, config.name, "final_checkpoint" ,"cumulative_motion.json"), 'r') as f:
                cumulative_motion=json.load(f)
                motion_cond = torch.tensor(cumulative_motion, dtype=torch.float16, device=distributed_state.device) # [1, 31]


    def collate_fn(data):
        pil_image = [example["pil_image"] for example in data]
        images = torch.stack([example["pixel_values"] for example in data])
        images = images.to(memory_format=torch.contiguous_format).float()
        texts = [example["text"] for example in data]
        img_ids = [example["img_id"] for example in data]

        return {
            "pil_image": pil_image,
            "pixel_values": images,
            "text": texts,
            "img_id": img_ids,
        }


    if config.test_data['dataset'] == "pororo":
        test_dataset = PororoDataset(**config.test_data)
    elif config.test_data['dataset'] == "flintstones":
        test_dataset = FlintstonesDataset(**config.test_data)

    print()
    if config.unet_additional_kwargs.first_frame_condition_mode=="none":
        print("Story Visualization")
    else:
        print("Story Continuation")
    print(f"Test dataset: {config.test_data['dataset']}")
    print(f"Test dataset size: {len(test_dataset)}")
    print()

    # DataLoaders creation:
    total_test_batch_size = config.test_batch_size * accelerator.num_processes
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=total_test_batch_size,
        num_workers=config.test_data.num_workers,
    )

    progress_bar = tqdm(
        range(0, len(test_dataloader)),
        initial=0,
        desc="Steps",
        position=0,
        leave=True,
        disable=not accelerator.is_local_main_process,
    )

    samples_seen = 0
    for step, batch in enumerate(test_dataloader):

        ori_images = list(chain(*batch["pil_image"]))
        texts = list(chain(*batch["text"]))
        img_ids = list(chain(*batch["img_id"]))

        with distributed_state.split_between_processes(batch["text"], apply_padding=True) as story_text, distributed_state.split_between_processes(batch["pixel_values"], apply_padding=True) as pixel_values:
            prompts = list(chain(*story_text))
            if config.unet_additional_kwargs.first_frame_condition_mode!="none":
                first_frames = pixel_values[:, 0, :, :, :]
            else:
                first_frames = None

            sample = pipeline(
                prompts,
                first_frames     = first_frames,
                negative_prompt       = None,
                num_inference_steps   = config.sampling_kwargs.steps,
                guidance_scale_txt    = config.sampling_kwargs.guidance_scale_txt,
                guidance_scale_img    = config.sampling_kwargs.guidance_scale_img,
                width                 = config.sampling_kwargs.width,
                height                = config.sampling_kwargs.height,
                video_length          = config.sampling_kwargs.n_frames,
                noise_sampling_method = config.unet_additional_kwargs.noise_sampling_method,
                noise_alpha           = float(config.unet_additional_kwargs.noise_alpha),
                eta                   = config.sampling_kwargs.ddim_eta,
                frame_stride          = config.sampling_kwargs.frame_stride,
                guidance_rescale      = config.sampling_kwargs.guidance_rescale,
                use_frameinit         = config.frameinit_kwargs.enable,
                frameinit_noise_level = config.frameinit_kwargs.noise_level,
                camera_motion         = config.frameinit_kwargs.camera_motion,
                motion_cond           = motion_cond,
            ).videos
            frames = []
            for cnt, samp in enumerate(sample):
                frames.extend(save_videos_grid(samp.unsqueeze(0), format="pil"))

            gather_new_images = accelerator.gather_for_metrics(frames)

            if accelerator.is_main_process:
                initial_idx = step*total_test_batch_size
                save_generated_images(ori_images, gather_new_images, texts, img_ids, test_result_dir, initial_idx, video_length=config.sampling_kwargs.n_frames)

        accelerator.wait_for_everyone()
        progress_bar.update(1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/training/training.yaml")
    parser.add_argument("--name", "-n", type=str, default="")
    args = parser.parse_args()

    name   = args.name + "_" + Path(args.config).stem

    config = OmegaConf.load(args.config)
    config.name = name

    main(config)
