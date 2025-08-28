from torch.utils.data import Dataset
from scene.cameras import Camera
import torch
from PIL import Image

from torchvision import transforms as T
import numpy as np
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        args,
        dataset_type,
    ):
        self.dataset = dataset
        self.args = args
        self.dataset_type=dataset_type
        self.transform = T.ToTensor()
        
        
    def __getitem__(self, index):
        # TODO Create data loading function
        R = self.dataset[index].R
        T = self.dataset[index].T
        FovX = self.dataset[index].FovX
        FovY = self.dataset[index].FovY
        time = self.dataset[index].time
        
        img = Image.open(self.dataset[index].image_path).convert("RGB")
        
        # Downsample
        scale = 0.5  # <-- change this factor as needed
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img = self.transform(img)
        
        scale = 0.5  # <-- change this factor as needed
        mask = Image.open(self.dataset[index].image_path.replace('train', 'cropped').replace('jpg', 'png')).convert("RGB")
        mask = mask.resize((new_w, new_h), Image.LANCZOS)
        mask = self.transform(mask)
        
        return Camera(colmap_id=index, R=R, T=T, FoVx=FovX, FoVy=FovY, image=img, gt_alpha_mask=None,
                        image_name=f"{index}", uid=index, data_device=torch.device("cuda"), time=time, 
                        mask=mask, 
                        depth=None
                        )

    def __len__(self):
        
        return len(self.dataset)
