import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from imgaug import augmenters as iaa
import cv2
#
import warnings
warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="torchvision")

class HerosDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = self.load_images()

        self.heros_aug = iaa.Sometimes(
            0.5,
            iaa.Sequential([iaa.GaussianBlur(sigma=(1, 5.0)),
                            iaa.Affine(scale=(0.2, 0.8))]),
            iaa.Sequential([iaa.Affine(scale=(1.0, 1.5)),
                            iaa.AverageBlur(k=(3, 11))]),
            )
        # self.heros_aug = iaa.Sometimes(
        #     0.5,
        #     iaa.Sequential([iaa.BlendAlpha(
        #             (0.0, 1.0),
        #             foreground=iaa.MedianBlur(11),
        #             per_channel=True
        #         )]),
        #     iaa.Sequential(iaa.BlendAlphaSimplexNoise(
        #         foreground=iaa.BlendAlphaSimplexNoise(
        #             foreground=iaa.EdgeDetect(1.0),
        #             background=iaa.LinearContrast((0.5, 2.0)),
        #             per_channel=True
        #         ),
        #         background=iaa.BlendAlphaFrequencyNoise(
        #             exponent=(-2.5, -1.0),
        #             foreground=iaa.Affine(
        #                 rotate=(-10, 10),
        #                 translate_px={"x": (-4, 4), "y": (-4, 4)}
        #             ),
        #             background=iaa.AddToHueAndSaturation((-40, 40)),
        #             per_channel=True
        #         ),
        #         per_channel=True,
        #         aggregation_method="max",
        #         sigmoid=False
        #     )),
        #     )
        self.transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

        ]
        )

    def load_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for filename in os.listdir(class_path):
                image_path = os.path.join(class_path, filename)
                images.append((image_path, class_idx))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, class_idx = self.images[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.heros_aug(image = img)
        img = self.transform(img)

        return img, class_idx