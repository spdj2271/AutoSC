# Deep Spectral Clustering via Autoregressive Assignment Generation (AutoSC)

Official implementation of AutoSC, an autoregressive spectral clustering framework that models cluster assignments as a sequence and learns them with a Transformer.

AutoSC reformulates spectral clustering as a sequential decision process, enabling end-to-end learning of cluster assignments while preserving the spectral clustering objective.


## Pretrained Features

For convenience, we provide pre-extracted representations using a pretrained ViT-B/32 backbone.

Download features:

https://drive.google.com/file/d/1iOKLe1WJGRawemSDxljbM6pAe0HprdQ7/view?usp=sharing

These features are extracted using the pretrained CLIP ViT-B/32 encoder (https://huggingface.co/openai/clip-vit-base-patch32) and are used as the input representations for clustering experiments.


After downloading:
```
wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1iOKLe1WJGRawemSDxljbM6pAe0HprdQ7" -O dataset_embedding.tar.gz
tar -xzf dataset_embedding.tar.gz
```

The directory should look like:
```
AutoSC
└── data_HFCLIP
    ├── ImageNet-10_image_embedding_train.npy
    └── ImageNet-10_labels_train.txt
├── AutoSC.py
├── utils_AutoSC.py
├── dataset.py
├── HFCLIP_Feature_Extract.py
├── README.md
└── requirements.txt
```
