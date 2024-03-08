import pandas as pd
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import cv2
import json

def get_active_class(row, column_to_id):
    active_column = row[row == 1].index[0]
    # Get class ID from mapping
    return column_to_id.get(active_column)

def simplify_ground_truth(gt_file, mapping_file):
    data = pd.read_csv(gt_file)
    with open(mapping_file, "r") as f:
        column_to_id = json.load(f)

    results = []
    for _, row in data.iterrows():
        class_id = get_active_class(row, column_to_id)
        if class_id:  # Check if ID exists (avoiding None values)
            results.append({"image": row[0], "class_id": class_id})

    return results

class ImageLabelDataset(Dataset):
  def __init__(self, image_folder, gt_file, mapping_file, img_size):
    self.image_folder = image_folder
    self.labels = simplify_ground_truth(gt_file, mapping_file)
    self.transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(img_size), 
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, saturation=0.2),
                        transforms.RandomRotation(degrees=90),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        ])

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    # Get image path based on index (assuming filenames correspond to labels)
    image_path = f"{self.image_folder}/{self.labels[idx]['image']}.jpg"  # Modify extension if needed
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = self.labels[idx]["class_id"]
    img_name = self.labels[idx]['image']

    # Apply transformations
    image_tensor = self.transform(image)

    return image_tensor, label, img_name

def make_dataset(image_folder, gt_file, mapping_file, img_size = (224, 224), test_size = 0.2):
   dataset = ImageLabelDataset(image_folder, gt_file, mapping_file, img_size)
   test_datapoints= int(len(dataset) * test_size)
   train_dataset, test_dataset = random_split(dataset, [len(dataset) - test_datapoints, test_datapoints])
   return {"train": train_dataset, 
           "test" : test_dataset}

if __name__ == "__main__":
    image_folder = r"data/KCDH2024_Training_Input_10K"
    gt_file = r"data/KCDH2024_Training_GroundTruth.csv"
    mapping_file = r"disease_id.json"
    data = make_dataset(image_folder, gt_file, mapping_file)
    