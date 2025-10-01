import os, pickle
from tqdm import tqdm
import numpy as np
import torch.utils.data
import PIL
from PIL import Image
from random import randrange
import random
from collections import Counter
import json
import torchvision.transforms as transforms

unique_characters = ["Wilma", "Fred", "Betty", "Barney", "Dino", "Pebbles", "Mr Slate"]
female = ["Wilma", "Betty", "Pebbles"]
all_characters = [ "fred", "barney", "wilma", "betty", "pebbles", "dino", "slate"]


class FlintstonesDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder="/ssd/zhengsixiao/datasets/flintstones_data", mode='train',  min_len=4,  sample_size=256, t_drop_rate=0.1, **kwargs):
        super().__init__()
        self.followings = {}
        self.im_input_size = sample_size
        self.data_folder = data_folder
        self.t_drop_rate = t_drop_rate
        self.mode = mode
        self.min_len = min_len

        splits = json.load(open(os.path.join(data_folder, 'train-val-test_split.json'), 'r'))
        train_id, val_id, test_id = splits["train"], splits["val"], splits["test"]

        if os.path.exists(os.path.join(data_folder, 'following_cache' + str(min_len) +  '.pkl')):
            self.followings = pickle.load(open(os.path.join(data_folder, 'following_cache' + str(min_len) +  '.pkl'), 'rb'))
        else:
            all_clips = train_id + val_id + test_id
            all_clips.sort()
            for idx, clip in enumerate(tqdm(all_clips, desc="Counting total number of frames")):
                season, episode = int(clip.split('_')[1]), int(clip.split('_')[3])
                has_frames = True
                for c in all_clips[idx+1:idx+min_len+1]:
                    s_c, e_c = int(c.split('_')[1]), int(c.split('_')[3])
                    if s_c != season or e_c != episode:
                        has_frames = False
                        break
                if has_frames:
                    self.followings[clip] = all_clips[idx+1:idx+min_len+1]
                else:
                    continue
            pickle.dump(self.followings, open(os.path.join(data_folder, 'following_cache' + str(min_len) + '.pkl'), 'wb'))

        ### character
        if os.path.exists(os.path.join(data_folder, 'labels.pkl')):
            self.labels = pickle.load(open(os.path.join(data_folder, 'labels.pkl'), 'rb'))
        else:
            print("Computing and saving labels")
            annotations = json.load(open(os.path.join(data_folder, 'flintstones_annotations_v1-0.json'), 'r'))
            self.labels = {}
            for sample in annotations:
                sample_characters = [c["entityLabel"].strip().lower() for c in sample["characters"]]
                self.labels[sample["globalID"]] = [1 if c.lower() in sample_characters else 0 for c in unique_characters]
            pickle.dump(self.labels, open(os.path.join(data_folder, 'labels.pkl'), 'wb'))

        ### description and backgorund (settings)
        self.descriptions = {}
        self.settings = {}
        self.all_settings = []
        self.characters = {}
        annotations = json.load(open(os.path.join(data_folder, 'flintstones_annotations_v1-0.json'), 'r'))
        for sample in annotations:
            self.descriptions[sample["globalID"]] = sample["description"]
            self.settings[sample["globalID"]] = sample["setting"]
            self.characters[sample["globalID"]] = [c["entityLabel"].strip() for c in sample["characters"]]
            if not sample["setting"] in self.all_settings:
                self.all_settings.append(sample["setting"])

        train_id = [tid for tid in train_id if tid in self.followings and len(self.followings[tid]) == min_len]
        val_id = [vid for vid in val_id if vid in self.followings and len(self.followings[vid]) == min_len]
        test_id = [tid for tid in test_id if tid in self.followings and len(self.followings[tid]) == min_len]
        
        if mode == 'train':
            self.orders = train_id
            self.transform = transforms.Compose([
                transforms.Resize(self.im_input_size),
                # transforms.RandomHorizontalFlip(),
         		transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])

        elif mode =='val':
            self.orders = val_id
            self.transform = transforms.Compose([
                transforms.Resize(self.im_input_size),
         		transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
        elif mode == 'test':
            self.orders = test_id
            self.transform = transforms.Compose([
                transforms.Resize(self.im_input_size),
         		transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
        else:
            raise ValueError
    
    def sample_image(self, path):
        arr = np.load(path)
        n_frames = arr.shape[0]
        random_range = randrange(n_frames)
        im = arr[random_range]
        return im

    def __getitem__(self, item):
        globalIDs = [self.orders[item]] + self.followings[self.orders[item]]

        pil_image_list = []
        image_list = []
        text_list= []
        img_id_list=[]

        for idx, globalID in enumerate(globalIDs):
            path = os.path.join(self.data_folder, 'video_frames_sampled', globalID + '.npy')

            im = self.sample_image(path)
            raw_img = Image.fromarray(im.astype('uint8'), 'RGB')
            image = self.transform(raw_img)
            imidiate_char = self.characters[globalID]
            text = self.descriptions[globalID].lower()

            pil_image_list.append(raw_img)
            image_list.append(image)
            text_list.append(text)
            img_id_list.append(globalID)

        image_tensor = torch.stack(image_list,0)

        return {
            "pil_image": pil_image_list,
            "pixel_values": image_tensor,
            "text": text_list,
            'img_id': img_id_list,
        }

    def __len__(self):
        return len(self.orders)


if __name__=="__main__":
    from transformers import CLIPTextModel, CLIPTokenizer

    pretrained_model_name_or_path = "ali-vilab/text-to-video-ms-1.7b"
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", local_files_only=True)

    train_dataset = FlintstonesDataset(tokenizer=tokenizer, im_input_size=256)

    sample = next(iter(train_dataset))

    print(sample["image"].shape)
    print(sample["text_input_ids"].shape)
    print(sample["pil_image"])
    print(sample["story_text"])