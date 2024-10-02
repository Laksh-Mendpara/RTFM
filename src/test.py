import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np
import config

def test(dataloader, model):
    with torch.no_grad():

        model.eval()
        pred = torch.zeros(0, device=config.DEVICE)

        for i, input in enumerate(dataloader):   
            input = input.to(config.DEVICE)
            
            #why
            input = input.permute(0, 2, 1, 3)

            # score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            # scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input)
            
            score_abnormal, score_normal, \
                feat_select_abn, feat_sel_norm, feat_select_abn, \
                    feat_select_abn, logits, feat_select_abn, feat_select_abn, feat_mag = model(input)
            
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            
            sig = logits
            
            pred = torch.cat((pred, sig))

        #shanghai
        if config.DATASET == 'shanghai':
            gt = np.load('list/gt-sh.npy')
        
        #ucf
        else:
            gt = np.load('list/gt-ucf.npy')

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)

        #saving, printing and plotting stuff
        np.save(config.RESULT_DIR+'fpr.npy', fpr)
        np.save(config.RESULT_DIR+'tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)
        # print('auc : ' + str(rec_auc))

        precision, recall, th = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        np.save(config.RESULT_DIR+'precision.npy', precision)
        np.save(config.RESULT_DIR+'recall.npy', recall)
        
        return rec_auc


if __name__ == "__main__":
    from model import Model
    from dataset import ShanghaiDataset
    from torch.utils.data import DataLoader

    test_data = ShanghaiDataset(train=False)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    
    model = Model(config.FEATURE_SIZE, config.BATCH_SIZE).to(config.DEVICE)
    
    rec_acc = test(test_loader, model)
    print(rec_acc)
