import torch
import torch.nn.functional as F

def loss_fn_mir(reconstructions, original_images):
    return F.mse_loss(reconstructions, original_images)

def loss_fn_pm(similarity_scores, labels):
    return F.binary_cross_entropy(similarity_scores, labels)

def loss_fn_io(outputs, labels):
    return F.cross_entropy(outputs, labels)
