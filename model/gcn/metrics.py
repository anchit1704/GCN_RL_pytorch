import torch 


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = torch.nn.CrossEntropyLoss(preds,labels)
    mask = mask.type(torch.FloatTensor) 
    mask /= torch.mean(mask)
    loss *= mask
    return torch.mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = torch.eq(torch.argmax(preds, 1), torch.argmax(labels, 1))
    accuracy_all = correct_prediction.type(torch.FloatTensor)
    mask = mask.type(torch.FloatTensor)
    mask /= torch.mean(mask)
    accuracy_all *= mask
    return torch.mean(accuracy_all)