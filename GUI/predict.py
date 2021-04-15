from torch.jit import load
from torchvision.transforms import Normalize
from torchvision.transforms import Compose
from torchvision.transforms import Resize
from torchvision.transforms import CenterCrop
from torchvision.transforms import ToTensor
from PIL.Image import open
import sys
import os

class RsModel:
    def __init__(self):
        #self.net=load(resource_path(os.path.join("res","torch_script_eval.pt")))
        self.net=load("torch_script_eval.pt")
        normalize = Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transforms=Compose(
            [
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                normalize,
            ]
        )
        self.category="airplane airport baseball_diamond basketball_court beach bridge chaparral church circular_farmland cloud commercial_area dense_residential desert forest freeway golf_course ground_track_field harbor industrial_area intersection island lake meadow medium_residential mobile_home_park mountain overpass palace parking_lot railway railway_station rectangular_farmland river roundabout runway sea_ice ship snowberg sparse_residential stadium storage_tank tennis_court terrace thermal_power_station wetland".split()
    #path为输入图片的路径
    def predict(self,path):
        self.net.eval()
        input_image = open(path)
        input_image=self.transforms(input_image).unsqueeze(0)
        output=self.net(input_image)
        return self.category[output.argmax().item()]
#生成资源文件目录访问路径
"""
def resource_path(relative_path):
    if getattr(sys, 'frozen', False): #是否Bundle Resource
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
"""





