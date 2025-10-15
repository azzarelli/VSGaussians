from torch.utils.data import Dataset
from scene.cameras import Camera
import torch

from torchvision import transforms as T
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        dataset_type,
    ):
        self.dataset = dataset
        self.dataset_type = dataset_type
        
        self.loading_flags = {
            "image":True,
            "scene_occluded": False if dataset_type == 'canon' else True
            
        }
        
    def __getitem__(self, index):
        
        cam = Camera(
            R=self.dataset[index].R, T=self.dataset[index].T,
            fx=self.dataset[index].fx, fy=self.dataset[index].fy,
            cx=self.dataset[index].cx, cy=self.dataset[index].cy,
            
            width=self.dataset[index].width, height=self.dataset[index].height,

            time=self.dataset[index].time,

            image_path=self.dataset[index].image_path,
            sceneoccluded_path=self.dataset[index].so_path,
            uid=self.dataset[index].uid,
            data_device=torch.device("cuda"), 
        )
        
        if self.loading_flags["image"]:
            cam.load_image_from_flags("image")
            
        if self.loading_flags["scene_occluded"]:
            cam.load_image_from_flags("scene_occluded")
            
        return cam

    def __len__(self):
        
        return len(self.dataset)

from PIL import Image
class IBLBackround(Dataset):
    def __init__(
        self,
        dataset,
    ):
        self.dataset = dataset
        self.transform = T.ToTensor()
        self.abc = torch.rand(3,3)
        
    def update_abc(self, abc):
        self.abc = abc.detach()


        
    def __getitem__(self, index):
        image = self.transform(
            Image.open(self.dataset[index]).convert("RGB")
        )
        return image

    def __len__(self):
        
        return len(self.dataset)