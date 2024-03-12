import torch
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os

def get_list_from_csv(labels_file):
    with open(labels_file, newline='') as f:
        reader = csv.reader(f)
        labels = list(reader)

    return labels

class ImageLabelDataset(Dataset):
  def __init__(self, image_folder, labels_file, img_size):
    self.image_folder = image_folder
    self.labels = get_list_from_csv(labels_file)
    self.transform = transforms.Compose([
                        transforms.Resize(img_size), 
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        ])

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    # Get image path based on index (assuming filenames correspond to labels)
    img = self.labels[idx][0] + ".jpg"
    image_path = os.path.join(self.image_folder, img)
    #image_path = r"{}\{}.jpg".format(self.image_folder, self.labels[idx][0])  # Modify extension if needed
    image = Image.open(image_path)
    image = image.convert('RGB')
    img_name = self.labels[idx][0]

    # Apply transformations
    image_tensor = self.transform(image)

    return image_tensor, img_name

class DataUtility:
  def __init__(self, image_folder, labels_file, img_size = (224, 224)) -> None:
    self.image_folder = image_folder
    self.labels_file = labels_file
    self.img_size = img_size
    self.dataset = None
    self.dataloader = None

  def make_dataset(self):
    self.dataset = ImageLabelDataset(self.image_folder, self.labels_file, self.img_size)
    return self.dataset
  
  def make_dataloader(self, batch_size = 32):
     #Exception check for dataset creation
     if self.dataset is None:
        self.make_dataset()

     self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

     return self.dataloader
  
  def create(self, dataset = True, dataloader = True, batch_size = 32):
    if dataset:
      self.make_dataset()

    if dataloader:
       self.make_dataloader(batch_size)


def generate_predictions(model, DataClass, device): 
    model.eval()
    model.to(device)
    
    predictions = torch.tensor([], device=device)
    image_ids = []

    with torch.no_grad():
        for _, metadata in enumerate(DataClass.dataloader):
            inputs = metadata[0]
            inputs = inputs.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            predictions = torch.cat((predictions, predicted), dim = 0)
            image_ids += metadata[1]
        
    predictions = predictions.cpu()
    return predictions.numpy().astype(np.int8), image_ids

def make_NV_folders(model, params):
  DataClassEval = DataUtility(params['eval_image_folder'], params['eval_labels'], params['img_size'])
  DataClassEval.create(batch_size=batch_size)
  
  predictions, image_ids = generate_predictions(model, DataClassEval, device)
    
  yes_NV = os.path.join(params['output_folder_path'], 'NV')
  no_NV = os.path.join(params['output_folder_path'], 'not_NV')
  images_dir = os.path.join(params['eval_image_folder'], 'KCDH2024_Test_Input')
  os.makedirs(yes_NV, exist_ok=True)
  os.makedirs(no_NV, exist_ok=True)
    
  for image_id, prediction in zip(image_ids, predictions):
    source_path = os.path.join(images_dir, f'{image_id}.jpg')
    if prediction == 0:
      target_path = os.path.join(not_NV, f'{image_id}.jpg')
    else:
      target_path = os.path.join(NV, f'{image_id}.jpg')
        
    # Move the file
    shutil.move(source_path, target_path)
    

def make_submission_csv(params, model, device = "cpu", csv_file_path = 'submission.csv', batch_size = 32):
    DataClassEval = DataUtility(params['eval_image_folder'], params['eval_labels'], params['img_size'])
    DataClassEval.create(batch_size=batch_size)
    predictions, image_ids = generate_predictions(model, DataClassEval, device)
    submission_df = pd.DataFrame({'ID': image_ids, 'Class': predictions})
    submission_df.to_csv(csv_file_path, index=False)
