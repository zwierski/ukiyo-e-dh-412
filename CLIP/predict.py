import numpy as np
import pandas as pd
import torch
from pkg_resources import packaging
import clip
import os
#import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader


class image_text_dataset(Dataset):
    def __init__(self, list_image_paths,list_texts,preprocess):

        self.image_paths = list_image_paths
        self.texts  = clip.tokenize(list_texts) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.preprocess=preprocess
        self.original_texts = list_texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):            
      try:
        image = self.preprocess(Image.open(self.image_paths[idx]))  # Image from PIL module
        title = self.texts[idx]
        path = self.image_paths[idx]
        original_title=self.original_texts[idx]
      except:
        image = self.preprocess(Image.open(self.image_paths[0]))
        title = self.texts[0]
        path = self.image_paths[0]
        original_title=self.original_texts[0]
        
      return image,title,path,original_title

if __name__ == "__main__":
    print("Torch version:", torch.__version__)

    device = "cuda:1" if torch.cuda.is_available() else "cpu" 

    model, preprocess = clip.load("ViT-B/32",device=device,jit=False) #Must set jit=False for training
    model.cuda().eval()

    checkpoint = torch.load("/root/ukiyo-e/CLIP/best_model.pt")

    # Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
    #max_length = model.context_length
    max_length=77

    model.load_state_dict(checkpoint['model_state_dict'])

    # Load data
    data = pd.read_csv("/root/ukiyo-e/CLIP/predict_data.csv")
    data.dropna(subset=['Image URL'], inplace=True)
    data['LABEL'] = data['LABEL'].apply(lambda x: '' if isinstance(x,float) else x)
    data['LABEL'] = data['LABEL'].apply(lambda x: x[:70] if len(x)>max_length else x)

    image_paths = data['Image URL'].apply(lambda x: "/root/ukiyo-e/images/images/"+x.split('/')[-1]).tolist()
    texts = data['LABEL'].tolist()

    dataset = image_text_dataset(image_paths,texts,preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Get the image and text embedding
    text_prompts = ['Emperor', 'Shogun', 'Samurai', 'Minister', 'Constitution',
              'Courtroom', 'Contract', 'Patent', 'Judge', 'Lawyer', 'Police', 'Prison',
              'School', 'Uniform', 'Textbook', 'Scientific instrument',
              'Steamship', 'Telegraph', 'Brick', 'Bank', 'Factory', 'Steam', 'Worker', 'Railway', 'Train', 'Lantern', 'Bulb', 'Bus', 'Clock', 'Bicycle', 'Motorcycle', 'Car', 'Bridge',
              'Soldier', 'Gun', 'Warship', 'Sword', 'Army',
              'Kimono', 'Suit', 'Gown', 'Glove',
              'Farmer','Merchant', 'Craftsman', 'Kabuki',
              'Mountain', 'Sea', 'Lake', 'Plant', 'Animal', 'Man', 'Woman']
    text_prompts = text_prompts
    text_prompts = list(set(text_prompts))
    # index to label dictionary
    idx2label = {idx:label for idx,label in enumerate(text_prompts)}


    # create an empty dataframe to store the results
    results = pd.DataFrame(columns=['image_path','text','top_probs','top_labels'])

    batch_count = 0
    total_batch = len(dataloader)
    for image,text,path,original_text in dataloader:
        batch_count += 1
        print(f"Batch {batch_count} out of {total_batch}")
        with torch.no_grad():
            # get the image and text embedding
            #text = clip.tokenize(text)
            image_features = model.encode_image(image.cuda())
            text_features = model.encode_text(text.cuda())

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # use the image embedding to get the top 5 text embedding from the text prompts
            prompt_descriptions = [f"This is a photo of a {label}" for label in text_prompts]
            prompt_tokens = clip.tokenize(prompt_descriptions).cuda()

            with torch.no_grad():
                prompt_features = model.encode_text(prompt_tokens).float()
                prompt_features /= prompt_features.norm(dim=-1, keepdim=True)

            prompt_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).float()
            try:       
                top_probs, top_labels = prompt_probs.cpu().topk(7, dim=-1)
                top_probs = top_probs.tolist()
                top_labels = top_labels.tolist()
                top_labels = [[idx2label[idx] for idx in labels] for labels in top_labels]
            except:
                continue

        # save the top_probs and top_labels       
        new_data = pd.DataFrame({'image_path': path, 'text': original_text, 'top_probs': list(top_probs), 'top_labels': list(top_labels)})
        results = pd.concat([results, new_data], ignore_index=True)
        #results = results.append({'image_path':path,'text':text,'top_probs':top_probs,'top_labels':top_labels},ignore_index=True)
    
    # save the results
    results.to_csv("results.csv",index=False)


    