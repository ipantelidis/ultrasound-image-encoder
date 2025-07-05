import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np

class MultiTaskDataset(Dataset):
    def __init__(self, df, task_type, sequence_length=4, transform=None):
        """
        :param df: Pandas DataFrame with a column 'path' containing image paths
        :param task_type: The type of task to load (e.g., 'MIR', 'PM', 'IO')
        :param sequence_length: The length of image sequences for 'IO' task
        :param transform: Optional transformations for image preprocessing
        """
        self.df = df
        self.task_type = task_type
        self.sequence_length = sequence_length
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image_path = self.df.iloc[index]['frame_path']
        image = Image.open(image_path).convert("L")

        if self.task_type == 'MIR':
            original_image = self.transform(image)
            masked_image = self.apply_random_mask(original_image)
            return masked_image, original_image
        
        elif self.task_type == 'PM':
            pair_image_path, label = self.get_patient_pair(image_path)
            image1 = self.transform(image)
            image2 = self.transform(Image.open(pair_image_path).convert("L"))
            return (image1, image2), torch.tensor(label, dtype=torch.float32)
        
        elif self.task_type == 'IO':
            sequence, correct_order = self.get_image_sequence(index)
            sequence = torch.stack([self.transform(Image.open(path).convert("L")) for path in sequence])
            return sequence, torch.tensor(correct_order, dtype=torch.long)
        
    def apply_random_mask(self, image, mask_ratio=0.3, num_patches=14):
        h, w, _ = image.shape 
        patch_size_h, patch_size_w = h // num_patches, w // num_patches

        mask = np.ones((num_patches, num_patches))
        num_mask = int(mask_ratio * num_patches ** 2)

        masked_indices = random.sample(range(num_patches ** 2), num_mask)
        for i in masked_indices:
            row, col = i // num_patches, i % num_patches
            mask[row, col] = 0

        masked_image = image.clone()
        for row in range(num_patches):
            for col in range(num_patches):
                if mask[row, col] == 0:
                    start_h, end_h = row * patch_size_h, (row + 1) * patch_size_h
                    start_w, end_w = col * patch_size_w, (col + 1) * patch_size_w
                    masked_image[start_h:end_h, start_w:end_w] = 0  

        return masked_image

    def get_patient_pair(self, image_path):
        video_name = self.df[self.df['frame_path'] == image_path]['video_name'].values[0]

        if random.random() > 0.5: # 50% to get a matching pair
            matching_images = self.df[self.df['video_name'] == video_name]['frame_path'].tolist()
            matching_images.remove(image_path)
            if matching_images:
                pair_image_path = random.choice(matching_images)
                label = 1
            else:
                pair_image_path = self.df[self.df['video_name'] != video_name].sample(1)['frame_path'].values[0]
                label = 0
        else:
            non_matching_images = self.df[self.df['video_name'] != video_name]['frame_path'].tolist()
            pair_image_path = random.choice(non_matching_images)
            label = 0

        return pair_image_path, label

    def get_image_sequence(self, index):
        video_name = self.df.iloc[index]['video_name']
        sequence_df = self.df[self.df['video_name'] == video_name].sort_values(by='frame_path')
        sequence_paths = sequence_df['frame_path'].tolist()

        if len(sequence_paths) < self.sequence_length:
            sequence_paths = (sequence_paths * ((self.sequence_length // len(sequence_paths)) + 1))[:self.sequence_length]
        else:
            sequence_paths = random.sample(sequence_paths, self.sequence_length)
        
        correct_order = list(range(len(sequence_paths)))
        shuffled_sequence = list(zip(sequence_paths, correct_order))
        random.shuffle(shuffled_sequence)

        sequence_paths, correct_order = zip(*shuffled_sequence)

        return list(sequence_paths), list(correct_order)
