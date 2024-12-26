from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2
import torch
from flatform.src.ultis import split_data

class MultiTaskDataset(Dataset):
    def __init__(self, datasets, size = 448, transforms=None):
        super().__init__()
        self.datasets = datasets
        self.size = size
    
        self.transforms = transforms
        
        self.transform_image = A.Compose(
            [
                A.Resize(height = self.size, width = self.size, interpolation = cv2.INTER_LINEAR),
                A.HorizontalFlip(p = 0.6),
                A.Blur(),
                A.RandomBrightnessContrast(p = 0.6),
                A.CoarseDropout(p = 0.6, max_holes=18, max_height=24, max_width=24, min_holes=12, min_height=12, min_width=12, fill_value=0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), 
                            max_pixel_value=255.0),
            ]
        )

    def __len__(self):
        return len(self.datasets)
    

    def __getitem__(self, index):
        label, mask_path, image_path = self.datasets[index]
        
        name = image_path.name.split(".")[0]
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            transformed = self.transform_image(image=image, mask=mask)
            transformed_image = transformed["image"]
            transfromed_mask = transformed["mask"]
        

        tensor_image = torch.from_numpy(transformed_image).permute(2,0,1).to(torch.float32)

        return label, tensor_image, transfromed_mask, name
    

def build_data(data_dir, imgsz, transforms, shuffle, batch_size, num_workers):
     process_data = split_data(data_dir)
     
     data = MultiTaskDataset(process_data, size=imgsz,
                        	transforms=transforms)
     
     data_loader = DataLoader(data, shuffle=shuffle,
                             batch_size=batch_size, num_workers=num_workers)
     
     return data_loader