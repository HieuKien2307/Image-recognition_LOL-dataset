import os
import io
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from models.convnext import convnext_base
import cv2
import matplotlib.pyplot as plt

 ## Detect function
def crop_heros_face(img):

    h, w, c = img.shape
    gray = cv2.cvtColor(img[:,:int(w/2),:], cv2.COLOR_RGB2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    #detected_circles
    detected_circles = cv2.HoughCircles(gray_blurred,
        cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 20, maxRadius= 80)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        x, y, r = detected_circles[0][0]
        y = int(y)
        crop_img = img[y-r:y+r, x-r:x+r,:]
        # print(crop_img.shape)
        return crop_img
    else:
        h, w, c = img.shape
        crop_img = img[:,int(w/8):int(w/2)-int(w/8),:]
        return crop_img

def get_model():
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = convnext_base(pretrained = False)

    state_dict = torch.load("checkpoints/Convnext_base_strong_aug_4_2023Nov28_15.51")
    model.load_state_dict(state_dict["net"])
    model.to(device)
    model.eval()

    ##Prepare data
    labels = os.listdir("datasets/train_data")

    return model, labels

def get_output(results, output_path):
    f = open(output_path, "a")
    for i in results:
        f.write(f"{i}")
    f.close()
    print("Completed")

def predict_test(data_path, model, labels):

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ]
        )

    results = []
    images_path = os.listdir(data_path)
    images_path.sort()
    for path in images_path:
        img = cv2.imread(f"{data_path}/{path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        crop_img = crop_heros_face(img)
        # debug
        # print(path)
        # print(crop_img.shape)
        # plt.imshow(crop_img)
        # plt.show()
        crop_img = transform(crop_img)
        pred = model(torch.unsqueeze(crop_img,0).to(device))
        pred = nn.Softmax(1)(pred)
        pred = labels[pred.argmax(1)]

        result = f"{path}\t{pred}\n"
        results.append(result)
    return results

def main(data_path, output_path):
    model, labels = get_model()
    results = predict_test(data_path, model, labels)
    get_output(results, output_path)


if __name__ == "__main__":

    data_path = "datasets/test_data/test_images"
    output_path = "predict.txt"

    main(data_path, output_path)