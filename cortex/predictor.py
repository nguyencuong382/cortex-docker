import torch
import boto3
import pickle
import requests

from PIL import Image
from io import BytesIO
from torchvision import transforms


# initialize boto session
session = boto3.Session(
    aws_access_key_id="",
    aws_secret_access_key='',
    region_name=''
)

# define the predictor
class PythonPredictor:
    def __init__(self, config):
        s3 = session.client('s3')
        s3.download_file(config['bucket'], config['key'], 'model.pkl')
        self.model = pickle.load(open('model.pkl', 'rb'))
        self.model.eval()

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor(), normalize]
        )

        self.labels = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
                       'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
                       'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier',
                       'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
                       'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
                       'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland',
                       'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu',
                       'staffordshire_bull_terrier', 'wheaten_terrier',  'yorkshire_terrier']

        self.device = config['device']

    def predict(self, payload):
        image = requests.get(payload["url"]).content
        img_pil = Image.open(BytesIO(image))
        img_tensor = self.preprocess(img_pil)
        img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            prediction = self.model(img_tensor)
        _, index = prediction[0].max(0)
        return self.labels[index]