import torch
import numpy as np
from torchvision import transforms
import os
import random
from PIL import Image

class PororoDataset(torch.utils.data.Dataset):

    def __init__(self, image_root_path="/ssd/zhengsixiao/datasets/pororo_png", use_filtered_dataset=True, mode="train", min_len=4, sample_size=512, t_drop_rate=0.1, **kwargs):
        super().__init__()

        self.size = sample_size
        self.t_drop_rate = t_drop_rate
        self.image_root_path = image_root_path
        self.mode = mode
        self.use_filtered_dataset = use_filtered_dataset
        self.min_len = min_len

        self.images = np.load(os.path.join(self.image_root_path,'img_cache' + str(min_len) + '.npy'), encoding='latin1')
        self.followings = np.load(os.path.join(self.image_root_path,'following_cache' + str(min_len) + '.npy'))
        if use_filtered_dataset: # and self.mode =='test'
            self.descriptions = np.load(os.path.join(self.image_root_path, 'descriptions_blip2.npy'), allow_pickle=True, encoding='latin1').item()
            print("Using filtered dataset")
        else:
            self.descriptions = np.load(os.path.join(self.image_root_path, 'descriptions.npy'), allow_pickle=True, encoding='latin1').item()
        train_id, val_id, test_id = np.load(os.path.join(self.image_root_path, "train_seen_unseen_ids.npy"), allow_pickle=True)
        
        if self.mode == 'train':
            self.ids = np.sort(np.concatenate((train_id, val_id)))
        elif self.mode =='val':
            self.ids = np.sort(val_id)
        elif self.mode =='test':
            self.ids = np.sort(test_id)
        else:
            raise ValueError

        if self.mode == 'train':
            self.transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(self.size, antialias=None),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.size, antialias=None),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))
     
    def __getitem__(self, item):
        src_img_id = self.ids[item]
        all_img_ids = [str(self.images[src_img_id].decode('utf-8'))[:-4]]
        tgt_img_ids = [str(self.followings[src_img_id][i].decode('utf-8'))[:-4] for i in range(self.min_len)]
        all_img_ids = all_img_ids + tgt_img_ids

        pil_image_list = []
        image_list = []
        text_list= []

        for idx, img_id in enumerate(all_img_ids):
            # read image
            if self.use_filtered_dataset: #  and self.mode =='test'
                src_img_path = os.path.join(self.image_root_path, "pororo_png_filtered_blip2", img_id+".png")
                raw_image = Image.open(src_img_path)
            else:
                src_img_path = os.path.join(self.image_root_path, img_id+".png")
                raw_image = self.sample_image(Image.open(src_img_path))
            image = self.transform(raw_image.convert("RGB"))

            # read text
            text = self.descriptions[img_id][0].lower()

            pil_image_list.append(raw_image.convert("RGB"))
            text_list.append(text)
            image_list.append(image)
            
        image_tensor = torch.stack(image_list,0)

        return {
            "pil_image": pil_image_list,
            "pixel_values": image_tensor,
            "text": text_list,
        }

    def __len__(self):
        return len(self.ids)