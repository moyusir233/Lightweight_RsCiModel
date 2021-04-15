from torch.jit import load
import torchvision.transforms as transforms
import PIL
class RsModel:
    def __init__(self):
        self.net=load("torch_script_eval.pt")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.transforms=transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.category="airplane airport baseball_diamond basketball_court beach bridge chaparral church circular_farmland cloud commercial_area dense_residential desert forest freeway golf_course ground_track_field harbor industrial_area intersection island lake meadow medium_residential mobile_home_park mountain overpass palace parking_lot railway railway_station rectangular_farmland river roundabout runway sea_ice ship snowberg sparse_residential stadium storage_tank tennis_court terrace thermal_power_station wetland".split()
    #path为输入图片的路径
    def predict(self,path):
        self.net.eval()
        input_image = PIL.Image.open(path)
        input_image=self.transforms(input_image).unsqueeze(0)
        output=self.net(input_image)
        return self.category[output.argmax().item()]
if __name__=="__main__":
    model=RsModel()
    print(model.predict("/home/igarss/NWPU-RESISC45/train/airplane/airplane_001.jpg"))




