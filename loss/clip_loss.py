import torch
from torch.nn import functional as F


def clip_forward(clip_model, images, tokenized_text):
    image_features = clip_model.encode_image(images)
    _, text_features = clip_model.encode_text(tokenized_text)

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    N, C = image_features.size()
    image_features = image_features.reshape(N, 1, C)
    N, C = text_features.size()
    text_features = text_features.reshape(N, C, 1)

    similarity = torch.matmul(image_features, text_features)

    return similarity

