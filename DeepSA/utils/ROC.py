from ssl import OP_NO_RENEGOTIATION
from unittest import result
import pandas as pd
import sys
import difflib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

if __name__ == "__main__":
    methods_list = [ "GASA", "sascore", "syba", "scscore", "rascore", "DeepSA_ChemMLM", "DeepSA_ChemMTR", "DeepSA_SmELECTRA", "DeepSA_MinBert", "DeepSAE_ChemMTR", "DeepSAE_ChemMLM" ]
    state_label = 'state'
    data_table = pd.read_csv(sys.argv[1], dtype={state_label:str})
    dataset_name = str(sys.argv[1].split(".")[0])
    postive_label = str(sys.argv[2]) # 1 for hs, 0 for es
    plt.figure(figsize=(10, 6), dpi=300)
    color_index=0
    for method in methods_list:
        if postive_label == '1':
            data_table['syba'] = -1*data_table['syba']
            data_table['rascore'] = -1*data_table['rascore']
            fpr, tpr, thersholds = roc_curve(data_table[state_label], data_table[method], pos_label = postive_label)
        elif postive_label == '0':
            data_table['sascore'] = -1*data_table['sascore']
            data_table['scscore'] = -1*data_table['scscore']
            fpr, tpr, thersholds = roc_curve(data_table[state_label], data_table[method], pos_label = postive_label)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='{} AUROC={:.3f}'.format(method, roc_auc))
        color_index+=1
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), markerfirst=False, frameon=True, framealpha=1)
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_statistics"+".png")
    plt.close()



