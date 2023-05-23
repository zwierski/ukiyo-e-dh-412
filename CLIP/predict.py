import numpy as np
import pandas as pd
import torch
from pkg_resources import packaging
import clip
import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader


class image_text_dataset(Dataset):
    def __init__(self, list_image_paths,list_texts):

        self.image_paths = list_image_paths
        self.texts  = clip.tokenize(list_texts) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_paths[idx])) # Image from PIL module
        title = self.texts[idx]
        path = self.image_paths[idx]
        return image,title,path

if __name__ == "__main__":
    print("Torch version:", torch.__version__)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    model.cuda().eval()

    checkpoint = torch.load("model_checkpoint/model_10.pt")

    # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
    max_length = model.context_length
    checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
    checkpoint['model_state_dict']["context_length"] =  max_length # default is 77
    checkpoint['model_state_dict']["vocab_size"] = model.vocab_size

    model.load_state_dict(checkpoint['model_state_dict'])
    max_length = 70

    # Load data
    data = pd.read_csv("CLIP/data.csv")
    data['Description'] = data['Description'].apply(lambda x: x[:70] if len(x)>max_length else x)

    image_paths = data['Image URL'].apply(lambda x: "images/images/"+x.split('/')[-1]).tolist()
    texts = data['Description'].tolist()

    dataset = image_text_dataset(image_paths,texts)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Get the image and text embedding
    text_prompts = ['Emperor', 'Shogun', 'Government officials', 'Voting', 'Constitution',
        'Courtroom', 'Legal document', 'Contracts', 'Patent', 'Judge', 'Lawyer','Police', 'Prison',
        'School', 'School uniform', 'Textbooks', 'Scientific instrument',
        'Steamship', 'Telegraph poles', 'Brick house',
        'Factory', 'Steam-powered machinery', 'Worker', 'Railway', 'Train', 'Train station',
        'Soldier', 'Gun', 'Military uniform', 'Warship',
        'Kimono', 'Western style clothing', 'Gloves']

    # create an empty dataframe to store the results
    results = pd.DataFrame(columns=['image_path','text','top_probs','top_labels'])

    for image,text,path in dataloader:
        with torch.no_grad():
            # get the image and text embedding
            text = clip.tokenize([f"This is a photo of a {des}" for des in text])
            image_features = model.encode_image(image.cuda()))
            text_features = model.encode_text(text.cuda()))

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # use the image embedding to get the top 5 text embedding from the text prompts
            prompt_descriptions = [f"This is a photo of a {label}" for label in text_prompts]
            prompt_tokens = clip.tokenize(prompt_descriptions).cuda()

            with torch.no_grad():
                prompt_features = model.encode_text(prompt_tokens).float()
                prompt_features /= prompt_features.norm(dim=-1, keepdim=True)

            prompt_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top_probs, top_labels = prompt_probs.cpu().topk(5, dim=-1)

        # save the top_probs and top_labels
        results = results.append({'image_path':path,'text':text,'top_probs':top_probs,'top_labels':top_labels},ignore_index=True)
    
    # save the results
    results.to_csv("results.csv",index=False)


    