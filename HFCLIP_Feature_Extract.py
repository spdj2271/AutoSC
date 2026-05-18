import os

import pandas as pd
from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", FutureWarning)

from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import numpy as np
# import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torchvision
from torchvision import datasets
from PIL import Image
import copy
from torchvision import transforms
from TAC import data_utils
from dataset_for_feature_extract import imagenet_10_cc, imagenet_dogs_cc, imagenet, imagenet_tiny
import torchvision.transforms.functional as F

dataset = ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "TinyImageNet",
           "DTD", "UCF-101", "ImageNet"][-2]


def get_ds_loaders_imagenet(dataset_name, transforms, batch_size, num_workers=8):
    # configure datasets according the given 'dataset_name'
    if dataset_name == 'ImageNet-10':
        path = "/data/wengang/dataset/imagenet"
        base_train, base_test = imagenet_10_cc(path, transforms=transforms)
        classes = 10
    elif dataset_name == 'ImageNet-Dogs':
        path = "/data/wengang/dataset/imagenet"
        base_train, base_test = imagenet_dogs_cc(path, transforms=transforms)
        classes = 15
    elif dataset_name == 'ImageNet':
        path = "/data/wengang/dataset/imagenet"
        base_train, base_test = imagenet(path, transforms=transforms)
        classes = 1000
    elif dataset_name == 'TinyImageNet':
        path = "/data/wengang/dataset/tiny-imagenet-200/train"
        base_train = ImageFolder(root=path, transform=transforms)
        base_test = ImageFolder(root=path, transform=transforms)
        classes = 200
    elif dataset == "DTD":
        train_set = datasets.DTD(
            root="/data/wengang/dataset/DTD",
            split="train",
            download=True,
            transform=transforms
        )
        val_set = datasets.DTD(
            root="/data/wengang/dataset/DTD",
            split="val",
            download=True,
            transform=transforms
        )
        base_train = ConcatDataset([train_set, val_set])

        base_test = datasets.DTD(
            root="/data/wengang/dataset/DTD",
            split="test",
            download=True,
            transform=transforms
        )
    elif dataset == "UCF-101":
        base_train = datasets.UCF101(
            root="/data/wengang/dataset/UCF101/UCF-101",  # 视频文件路径
            annotation_path="/data/wengang/dataset/UCF101/ucfTrainTestlist",
            frames_per_clip=1,
            step_between_clips=100000000,
            train=True,
            transform=transforms
        )
        base_test = datasets.UCF101(
            root="/data/wengang/dataset/UCF101/UCF-101",  # 视频文件路径
            annotation_path="/data/wengang/dataset/UCF101/ucfTrainTestlist",
            frames_per_clip=1,
            step_between_clips=100000000,
            train=False,
            transform=transforms
        )
        batch_size = 1
        num_workers = 0
    else:
        raise Exception(f"unknown dataset_name={dataset_name}")
    # dataset_train = LightlyDataset.from_torch_dataset(base_train, transform=transforms)
    # dataset_test = LightlyDataset.from_torch_dataset(base_test, transform=transforms)
    dataloader_train_ssl = torch.utils.data.DataLoader(base_train, batch_size=batch_size, shuffle=False,
                                                       pin_memory=False,
                                                       drop_last=False, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(base_test, batch_size=batch_size, shuffle=False, pin_memory=False,
                                                  drop_last=False, num_workers=num_workers)
    print(1)
    return dataloader_train_ssl, dataloader_test


def get_loaders(dataset, processor, input_size=224, batch_size=2048):
    class ClipVisionTransform:
        """
        Wrap a HuggingFace CLIPProcessor.feature_extractor into a callable transform
        that returns a torch.Tensor of shape (C, H, W), suitable for torchvision datasets.
        """

        def __init__(self, processor: CLIPProcessor):
            # processor.feature_extractor handles resizing, center crop, normalization, etc.
            self.feature_extractor = processor.feature_extractor

        def __call__(self, image: Image.Image) -> torch.Tensor:
            # feature_extractor returns dict with 'pixel_values' shaped (1, C, H, W)
            out = self.feature_extractor(images=image, return_tensors="pt")
            pv = out["pixel_values"]
            # remove batch dim
            return pv.squeeze(0)

    if dataset == "UCF-101":
        # transform = transforms.Compose([
        #     transforms.Resize(input_size),
        #     transforms.CenterCrop(input_size),
        #     # transforms.Lambda(lambda x: x.permute(0, 3, 1, 2).float() / 255.0),
        #     # transforms.Normalize(
        #     #     mean=[0.48145466, 0.4578275, 0.40821073],
        #     #     std=[0.26862954, 0.26130258, 0.27577711],
        #     # ),
        # ])
        def ucf_transform(video):
            # video: (T, H, W, C)

            center = video.shape[0] // 2
            frame = video[center]  # (H, W, C)

            # 转 (C, H, W)
            frame = frame.permute(2, 0, 1)
            frame = F.resize(frame, (224, 224))
            frame = frame.float() / 255.0
            return frame

        transform = ucf_transform
    else:
        transform = transforms.Compose([
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            ClipVisionTransform(processor)])
    if 'ImageNet' in dataset or 'DTD' in dataset or 'UCF-101' in dataset:
        dataloader_train, dataloader_test = get_ds_loaders_imagenet(dataset, transform,
                                                                    batch_size=batch_size)

    else:
        dataloader_train, dataloader_test = data_utils.get_dataloader(dataset, batch_size=batch_size,
                                                                      transforms=transform)
    return dataloader_train, dataloader_test


# Setting a global seed for reproducibility, Configuring GPU
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# pl.seed_everything(seed=0)
torch.use_deterministic_algorithms(True)
torch.multiprocessing.set_sharing_strategy('file_system')
logs_root_dir = os.path.join(os.getcwd(), "benchmark_logs")
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

model_name = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(model_name)

# model = CLIPModel.from_pretrained(model_name).vision_model.cuda()
model = CLIPModel.from_pretrained(model_name).cuda()
model.eval()

SIMPLE_IMAGENET_TEMPLATES = (
    lambda c: f"itap of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"art of the {c}.",
    lambda c: f"a photo of the small {c}.",
)


def get_prompt(words, index, device="cuda"):
    prompt = [SIMPLE_IMAGENET_TEMPLATES[index](word) for word in words]
    # text = clip.tokenize(prompt, truncate=True).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    text = tokenizer(prompt, padding=True, return_tensors="pt").to("cuda")
    return text


nouns = pd.read_csv("/home/wengang/code/SpectralFormer/TAC/data/WordNetNouns.csv").values
nouns_num = nouns.shape[0]
batch_size = 5120
for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
    features = []
    print("Inferring text features for index", index)
    for i in range(nouns_num // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > nouns_num:
            end = nouns_num
        nouns_batch = nouns[start:end]
        with torch.no_grad():
            prompt = get_prompt(nouns_batch[:, 0], index)
            feature = model.get_text_features(**prompt)
            features.append(feature.cpu().numpy())
        if i % 10 == 0:
            print(f"[Completed {i * batch_size}/{nouns_num}]")
    features = np.concatenate(features, axis=0)
    print("Feature shape:", features.shape)
    np.save("/home/wengang/code/SpectralFormer/data_HFCLIP512/nouns_embedding_prompt_" + str(index) + ".npy", features)
embeddings = np.zeros((nouns_num, 512))
for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
    embedding = np.load("/home/wengang/code/SpectralFormer/data_HFCLIP512/nouns_embedding_prompt_" + str(index) + ".npy")
    embeddings += embedding
embeddings = embeddings / len(SIMPLE_IMAGENET_TEMPLATES)
np.save("/home/wengang/code/SpectralFormer/data_HFCLIP512/nouns_embedding_ensemble.npy", embeddings)
# exit(0)

###########################      Image      ###########################
dataloader_train, dataloader_test = get_loaders(dataset, processor)

features = []
labels = []
print("Inferring image features and labels...")
for iteration, batch in enumerate(dataloader_train):
    print(f"iteration {iteration}")
    x = batch[0]
    if len(batch) == 3:
        y = batch[2]
    else:
        y = batch[1]
    x = x.cuda()
    if len(x.shape) == 5:
        # B, T, C, H, W = x.shape
        # video = x.view(B * T, C, H, W)
        # with torch.no_grad():
        #     frame_feat = model(video).pooler_output  # (B*T, 768)
        # frame_feat = frame_feat.view(B, T, -1)
        # feature = frame_feat.mean(dim=1)  # (B, 768)
        B, T, C, H, W = x.shape
        center = T // 2
        x = x[:, center]  # (B, C, H, W)
        feature = model(x).pooler_output
    else:
        with torch.no_grad():
            feature = model(x).pooler_output
    # if feature.shape[0] > 1:
    #     feature = feature[:1]
    features.append(feature.cpu().numpy())
    labels.append(y.numpy())
    # if iteration % 10 == 0:
    #     print(f"[Iter {iteration}/{len(dataloader_train)}]")
features = np.concatenate(features, axis=0)
labels = np.concatenate(labels, axis=0)
print("Feature shape:", features.shape, "Label shape:", labels.shape)

np.save("./data_HFCLIP/" + dataset + "_image_embedding_train.npy", features)
np.savetxt("./data_HFCLIP/" + dataset + "_labels_train.txt", labels)

features_test = []
labels_test = []
print("Inferring test image features and labels...")
for iteration, batch in enumerate(dataloader_test):
    print(f"iteration {iteration}")
    if len(batch) == 3:
        x, _, y = batch
    else:
        x, y = batch

    x = x.cuda()

    with torch.no_grad():

        if x.dim() == 5:  # 视频
            # B, T, C, H, W = x.shape
            # video = x.view(B * T, C, H, W)
            # frame_feat = model(video).pooler_output
            # frame_feat = frame_feat.view(B, T, -1)
            # feature = frame_feat.mean(dim=1)
            B, T, C, H, W = x.shape
            center = T // 2
            x = x[:, center]  # (B, C, H, W)
            feature = model(x).pooler_output
        else:
            feature = model(x).pooler_output
    # if feature.shape[0] > 1:
    #     feature = feature[:1]
    features_test.append(feature.cpu().numpy())
    labels_test.append(y.numpy())
    # if iteration % 10 == 0:
    #     print(f"[Iter {iteration}/{len(dataloader_test)}]")
# for iteration, (x, y) in enumerate(dataloader_test):
#     x = x.cuda()
#     with torch.no_grad():
#         feature = model(x).pooler_output
#     features_test.append(feature.cpu().numpy())
#     labels_test.append(y.numpy())
#     if iteration % 10 == 0:
#         print(f"[Iter {iteration}/{len(dataloader_test)}]")
features_test = np.concatenate(features_test, axis=0)
labels_test = np.concatenate(labels_test, axis=0)
print("Feature shape:", features_test.shape, "Label shape:", labels_test.shape)

if dataset == "CIFAR-20":
    coarse_label = [
        [72, 4, 95, 30, 55],
        [73, 32, 67, 91, 1],
        [92, 70, 82, 54, 62],
        [16, 61, 9, 10, 28],
        [51, 0, 53, 57, 83],
        [40, 39, 22, 87, 86],
        [20, 25, 94, 84, 5],
        [14, 24, 6, 7, 18],
        [43, 97, 42, 3, 88],
        [37, 17, 76, 12, 68],
        [49, 33, 71, 23, 60],
        [15, 21, 19, 31, 38],
        [75, 63, 66, 64, 34],
        [77, 26, 45, 99, 79],
        [11, 2, 35, 46, 98],
        [29, 93, 27, 78, 44],
        [65, 50, 74, 36, 80],
        [56, 52, 47, 59, 96],
        [8, 58, 90, 13, 48],
        [81, 69, 41, 89, 85],
    ]
    labels_copy = copy.deepcopy(labels)
    labels_test_copy = copy.deepcopy(labels_test)
    for i in range(20):
        for j in coarse_label[i]:
            labels[labels_copy == j] = i
            labels_test[labels_test_copy == j] = i

np.save("./data_HFCLIP/" + dataset + "_image_embedding_test.npy", features_test)
np.savetxt("./data_HFCLIP/" + dataset + "_labels_test.txt", labels_test)
