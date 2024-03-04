import os
import json

import cv2 as cv

from torch.utils.data import Dataset
from torchvision.io import read_image

from groundingdino.util.inference import load_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, transform=None, target_transform=None):
        self.labels_dir = annotations_dir
        self.img_dir = img_dir
        self.img_paths = [self.img_dir + "/" +  i for i in os.listdir(os.path.join(self.img_dir))]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        print("IMG_PATH: ", img_path)
        image_source, image= load_image(img_path)
        image_cv = cv.imread(img_path)
        label_file_object = open(os.path.join(self.labels_dir, img_path.split("/")[-1].replace(".jpg", ".json")))
        try:
            label_json = json.load(label_file_object)
            label = label_json["shapes"][0]["label"]
            points = label_json["shapes"][0]["points"]
        except:
            label = "not found"
            points = [[0, 0]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, image_source, image_cv, (label, points)