from sklearn.metrics import roc_auc_score

def freeze_model(model, exclude_layers):
    for i,(name, param) in enumerate(model.named_parameters()):
        requires_grad = False

        if i  in exclude_layers:
            requires_grad = True
        param.requires_grad = requires_grad


def computeAUROC(gt, pred):

    outAUROC = []

    gt = gt.cpu().numpy()
    pred = pred.cpu().numpy()

    for i in range(gt.shape[1]):

        outAUROC.append(roc_auc_score(gt[:, i], pred[:, i]))

    return outAUROC



def adjust_learning_rate_agent(optimizer, epoch,cfg):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    # if epoch >= 12:
    #     lr = cfg.lr_agent * 0.001
    # elif epoch >= 8.:
    #     lr = cfg.lr_agent * 0.01
    if epoch >= 4.:
        lr = cfg.lr_agent * 0.1
    else:
        lr = cfg.lr_agent

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr