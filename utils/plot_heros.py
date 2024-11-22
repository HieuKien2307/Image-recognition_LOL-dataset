import os
import numpy as np
import matplotlib.pyplot as plt




labels = os.listdir("/content/drive/MyDrive/AI_test/datasets/train_data")
# labels = sorted(labels)

# Tạo label map với số đứng trước
labels_map = {index: champion for index, champion in enumerate(labels)}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(custom_dataloader), size=(1,)).item()
    img, label = next(iter(custom_dataloader))
    image_np = np.transpose(img[0].squeeze().numpy(), (1, 2, 0))
    figure.add_subplot(rows, cols, i)
    label = label.item() if isinstance(label, torch.Tensor) else label
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(image_np)
plt.show()