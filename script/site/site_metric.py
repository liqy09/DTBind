import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, auc
import prettytable as pt
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, matthews_corrcoef, confusion_matrix, roc_auc_score


def eval_metrics(probs, targets, cal_AUC=True):
    threshold_list = [i / 50.0 for i in range(1, 50)]

    if cal_AUC:
        if isinstance(probs, torch.Tensor) and isinstance(targets, torch.Tensor):
            fpr, tpr, _ = roc_curve(y_true=targets.detach().cpu().numpy(), y_score=probs.detach().cpu().numpy())
            precision, recall, _ = precision_recall_curve(y_true=targets.detach().cpu().numpy(), probas_pred=probs.detach().cpu().numpy())
        elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
            fpr, tpr, _ = roc_curve(y_true=targets, y_score=probs)
            precision, recall, _ = precision_recall_curve(y_true=targets, probas_pred=probs)
        else:
            print('ERROR: probs or targets type is error.')
            raise TypeError
        auc_ = auc(x=fpr, y=tpr)
        auprc = auc(x=recall, y=precision)
    else:
        auc_ = 0
        auprc = 0

    threshold_best, rec_best, pre_best, F1_best, spe_best, mcc_best, acc_best, pred_bi_best = 0, 0, 0, 0, 0, -1, 0, None
    for threshold in threshold_list:
        metrics = th_eval_metrics(threshold, probs, targets, cal_AUC=False)
        if metrics['mcc'] > mcc_best:
            threshold_best, rec_best, pre_best, F1_best, spe_best, mcc_best, acc_best, pred_bi_best = (
                metrics['threshold'], metrics['rec'], metrics['pre'], metrics['f1'], metrics['spe'], metrics['mcc'], metrics['acc'], metrics['pred_bi']
            )

    return {
        'threshold': threshold_best,
        'acc': acc_best,
        'rec': rec_best,
        'pre': pre_best,
        'f1': F1_best,
        'spe': spe_best,
        'mcc': mcc_best,
        'auc': auc_,
        'auprc': auprc,
        'pred_bi': pred_bi_best
    }


def th_eval_metrics(threshold, probs, targets, cal_AUC=True):
    if isinstance(probs, torch.Tensor) and isinstance(targets, torch.Tensor):
        if cal_AUC:
            fpr, tpr, _ = roc_curve(y_true=targets.detach().cpu().numpy(), y_score=probs.detach().cpu().numpy())
            precision, recall, _ = precision_recall_curve(y_true=targets.detach().cpu().numpy(), probas_pred=probs.detach().cpu().numpy())
            auc_ = auc(x=fpr, y=tpr)
            auprc = auc(x=recall, y=precision)
        else:
            auc_ = 0
            auprc = 0
        pred_bi = targets.data.new(probs.shape).fill_(0)
        pred_bi[probs > threshold] = 1
        targets[targets == 0] = 5
        targets[targets == 1] = 10
        tn = torch.where((pred_bi + targets) == 5)[0].shape[0]
        fp = torch.where((pred_bi + targets) == 6)[0].shape[0]
        fn = torch.where((pred_bi + targets) == 10)[0].shape[0]
        tp = torch.where((pred_bi + targets) == 11)[0].shape[0]
        if tp > 0:
            rec = tp / (tp + fn)
        else:
            rec = 0
        if tp > 0:
            pre = tp / (tp + fp)
        else:
            pre = 0
        if tn > 0:
            spe = tn / (tn + fp)
        else:
            spe = 0
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0:
            mcc = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).item()
        else:
            mcc = 0  # 避免除以零的情况
        acc = (tp + tn) / (tp + tn + fp + fn)

    elif isinstance(probs, np.ndarray) and isinstance(targets, np.ndarray):
        fpr, tpr, _ = roc_curve(y_true=targets, y_score=probs)
        precision, recall, _ = precision_recall_curve(y_true=targets, probas_pred=probs)
        auc_ = auc(x=fpr, y=tpr)
        auprc = auc(x=recall, y=precision)

        pred_bi = np.abs(np.ceil(probs - threshold))

        tn, fp, fn, tp = confusion_matrix(targets, pred_bi).ravel()
        if tp > 0:
            rec = tp / (tp + fn)
        else:
            rec = 1e-8
        if tp > 0:
            pre = tp / (tp + fp)
        else:
            pre = 1e-8
        if tn > 0:
            spe = tn / (tn + fp)
        else:
            spe = 1e-8
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0:
            mcc = matthews_corrcoef(targets, pred_bi)
        else:
            mcc = 0  # 避免除以零的情况
        acc = (tp + tn) / (tp + tn + fp + fn)
        if rec + pre > 0:
            F1 = 2 * rec * pre / (rec + pre)
        else:
            F1 = 0
    else:
        print('ERROR: probs or targets type is error.')
        raise TypeError

    return {
        'threshold': threshold,
        'rec': rec,
        'pre': pre,
        'f1': F1,
        'spe': spe,
        'mcc': mcc,
        'auc': auc_,
        'auprc': auprc,
        'acc': acc,
        'pred_bi': pred_bi
    }


def CFM_eval_metrics(CFM):
    CFM = CFM.astype(float)
    tn = CFM[0, 0]
    fp = CFM[0, 1]
    fn = CFM[1, 0]
    tp = CFM[1, 1]
    if tp > 0:
        rec = tp / (tp + fn)
    else:
        rec = 0
    if tp > 0:
        pre = tp / (tp + fp)
    else:
        pre = 0
    if tn > 0:
        spe = tn / (tn + fp)
    else:
        spe = 0
    if rec + pre > 0:
        F1 = 2 * rec * pre / (rec + pre)
    else:
        F1 = 0
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0:
        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    else:
        mcc = -1
    return rec, pre,F1, spe, mcc

