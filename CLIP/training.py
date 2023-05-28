import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from pkg_resources import packaging
import clip
import os
import click
import IPython.display
import matplotlib.pyplot as plt
import random
import pandas as pd
from PIL import Image, ImageFile
import numpy as np

from collections import OrderedDict
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(20)




class image_text_dataset(Dataset):
    def __init__(self, list_image_path, list_txt,preprocess):
        self.image_path = list_image_path
        self.text = clip.tokenize(list_txt)  # you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.
        self.preprocess=preprocess

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
      try:
        image = self.preprocess(Image.open(self.image_path[idx]))  # Image from PIL module
        text = self.text[idx]
      except:
        image = self.preprocess(Image.open(self.image_path[0]))
        text = self.text[0]
        
      return image, text
        



def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


# Load data
def load_data(csv_data_address,batch_size,max_length,preprocess):
  data = pd.read_csv(csv_data_address)  
  data.dropna(subset=['LABEL', 'Image URL'], inplace=True)
  #data=data.sample(n=10000, random_state=42)
  data['LABEL'] = data['LABEL'].apply(lambda x: x[:70] if not isinstance(x,float) and len(x)>max_length else x)
  image_paths = data['Image URL'].apply(lambda x: "/root/ukiyo-e/images/images/"+x.split('/')[-1]).tolist()
  texts = data['LABEL'].tolist()  
  
  dataset = image_text_dataset(image_paths,texts,preprocess)
  dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)
  return dataloader

def train(csv_data_address,batch_size,epoch_num,max_length):  
  print("Torch version:", torch.__version__)
  device = "cuda:0" if torch.cuda.is_available() else "cpu" 
  
  model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
  if device == "cpu":
    model.float()
  else:
    clip.model.convert_weights(model)  # Actually this line is unnecessary since clip by default already on float16
    
  dataloader=load_data(csv_data_address,batch_size,max_length,preprocess)
  
  loss_img = nn.CrossEntropyLoss()
  loss_txt = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)  
  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

  best_loss = 1000000

  for epoch in range(epoch_num):
    print("Epoch now is: "+str(epoch))
    for batch in dataloader:
      try:
        optimizer.zero_grad()

        images, texts = batch

        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        print(total_loss)
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
      except:
        continue
    # save the best model
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': total_loss,
              }, f"best_model.pt") #just change to your preferred folder/filename
    print("Epoch now is finished, and best loss now is: "+str(best_loss))

@click.command()
@click.option('--csv_data_address', default="./data.csv", help='Address for the training data in csv form')
@click.option('--batch_size', default=36, help='Size of batch')
@click.option('--epoch_num', default=7, help='Number of epochs for training')
@click.option('--max_length', default=77, help='Maximum length of the description')
def train_command(csv_data_address,batch_size,epoch_num,max_length):
    return train(csv_data_address,batch_size,epoch_num,max_length) 

        
if __name__ == "__main__":    
    train_command()        


















    