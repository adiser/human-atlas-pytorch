import torch

def f1_micro(y_true, y_preds, thresh=0.5, eps=1e-20):
    preds_bin = y_preds > thresh # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true
    
    p = truepos.sum() / (preds_bin.sum() + eps) # take sums and calculate precision on scalars
    r = truepos.sum() / (y_true.sum() + eps) # take sums and calculate recall on scalars
    
    f1 = 2*p*r / (p+r+eps) # we calculate f1 on scalars
    return f1

def f1_macro(y_true, y_preds, thresh=0.5, eps=1e-20):
    preds_bin = y_preds > thresh # binary representation from probabilities (not relevant)
    truepos = preds_bin * y_true

    p = truepos.sum(axis=0) / (preds_bin.sum(axis=0) + eps) # sum along axis=0 (classes)
                                                            # and calculate precision array
    r = truepos.sum(axis=0) / (y_true.sum(axis=0) + eps)    # sum along axis=0 (classes) 
                                                            #  and calculate recall array

    f1 = 2*p*r / (p+r+eps) # we calculate f1 on arrays
    return np.mean(f1)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = torch.nn.functional.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()

"""
Smooth F2 Loss. Taken from github.com/zhijundeng/Amazon_Forest_Computer_Vision
"""

def torch_f2_score(y_true, y_pred):
    return torch_fbeta_score(y_true, y_pred, 2)

def torch_fbeta_score(y_true, y_pred, beta, eps=1e-9):
    beta2 = beta**2

    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))


class SmoothF2Loss(torch.nn.Module):
    def __init__(self):
        super(SmoothF2Loss, self).__init__()
    
    def forward(self, input, target):
        return 1 - torch_f2_score(target, torch.sigmoid(input))