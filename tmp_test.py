import numpy as np
import torch
from sklearn.metrics import f1_score
def overall_ordering_and_checking(pred_output, gt, proportion2=0.1,proportion1 = 0.5):
    """
    Overall ordering according to class 2 value. According to proportion given in the parameters, this function will
    classify all nodes into different class with fixed proportion.

    :param proportion: default is 10% -> class2 10% -> class1 80% -> class 0
    :param pred_output: Output of the model it should be converted from tensor to list first
    :param gt: ground truth of the label
    :return: it will return a list of nodes that belong to class 2, an accuracy only related to class2 and an overall F1
    """
    pred_output = pred_output.numpy()
    #pred_output_2 = pred_output[:,2]
    id_output_dict_2 = {}
    id_output_dict_1 = {}
    for i ,node_vec in enumerate(pred_output):
        id_output_dict_1[i] = id_output_dict_2[i] = node_vec
    id_output_dict_2 = sorted(id_output_dict_2.items(), key=lambda x: (x[1][2],x[1][1],x[1][0]), reverse=True)
    id_output_dict_1 = sorted(id_output_dict_1.items(), key=lambda x: (x[1][1], x[1][2], x[1][0]), reverse=True)
    pred_order = np.zeros(len(pred_output))
    cut_off_value2 = int(proportion2 * len(pred_output))
    cut_off_value1 = int(proportion1 * len(pred_output))
    for i,ele in enumerate(id_output_dict_2):
        if i <= cut_off_value2:
            pred_order[ele[0]] = 1
    #for i,ele in enumerate(id_output_dict_1):
    #    if i <= cut_off_value1:
    #        pred_order[ele[0]] = 1
    gt = gt.numpy()
    #print(pred_order==2)
    #print((pred_order == 2) & (gt == 2))
    acc_2 = sum((pred_order == 1) & (gt == 1)) / len(gt)
    f1 = f1_score(pred_order,gt,average='macro')
    print(pred_order,acc_2,f1)
    return pred_order,acc_2,f1


if __name__ == '__main__':
    pre = torch.tensor(np.array([[0.1,0.2,0.7],[0.2,0.3,0.5],[0.3,0.4,0.3],[0.6,0.1,0.3],[0.1,0.8,0.1],[0.1,0.7,0.2]]))
    gt = torch.tensor(np.array([2,2,1,0,1,0]))
    overall_ordering_and_checking(pre,gt)