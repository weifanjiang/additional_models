import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score
import os
from xgbKantchelianAttack import main
from sklearn.datasets import load_svmlight_file
import pickle
import os
import pandas as pd
import numpy as np

def eval(y, y_p):
    try:
        tn, fp, fn, tp = confusion_matrix(y, y_p).ravel()
        acc = (tp+tn)/float(tp+tn+fp+fn)
        fpr = fp/float(fp+tn)
        return acc, fpr
    except ValueError:
        return accuracy_score(y, y_p), None

datasets = ["breast_cancer", "binary_mnist", "cod-rna", "ijcnn"]
methods = ['exact', 'robust_exact', 'robust_centergreedy']
eps_vals = {
    "breast_cancer": {
        "robust_exact": 0.3,
        "robust_centergreedy": 0.3
    },
    "binary_mnist": {
        "robust_exact": 0.3,
        "robust_centergreedy": 0.3
    },
    "cod-rna": {
        "robust_exact": 0.2,
        "robust_centergreedy": 0.03
    },
    "ijcnn": {
        "robust_exact": 0.2,
        "robust_centergreedy": 0.02
    }
}
tree_size = {
    "breast_cancer": {
        "exact": (4, 6),
        "robust_exact": (4, 4),
        "robust_centergreedy": (2, 7)
    },
    "binary_mnist": {
        "exact": (600, 4),
        "robust_exact": (600, 8),
        "robust_centergreedy": (600, 9)
    },
    "cod-rna": {
        "exact": (40, 10),
        "robust_exact": (40, 4),
        "robust_centergreedy": (40, 7)
    },
    "ijcnn": {
        "exact": (80, 5),
        "robust_exact": (80, 10),
        "robust_centergreedy": (80, 10)
    }
}

attack_num = {
    "breast_cancer": 100,
    "binary_mnist": 100,
    "cod-rna": 5000,
    "ijcnn": 100
}

n_feat = {
    "binary_mnist": 784,
    "breast_cancer": 10,
    "cod-rna": 8,
    "ijcnn": 22
}

zero_based = {
    "binary_mnist": True,
    "breast_cancer": False,
    "cod-rna": True,
    "ijcnn": False
}

data_path = {
    "cod-rna": ("../DevRobustTrees/data/cod-rna.train.csv.round6", "../DevRobustTrees/data/cod-rna.test.csv.round6")
}

model_dir = 'models'
output_dir = 'log'

log_fname = 'GBDT_test_aacc.csv'
if os.path.isfile(log_fname):
    log_file = open(log_fname, "a+")
else:
    log_file = open(log_fname, "w")
    log_file.write("type,dataset,method,eps,nt,d,test acc,fpr\n")

for dataset in ["cod-rna"]:

    _, test_path = data_path[dataset]
    df_test = pd.read_csv(test_path, header=None)
    test = df_test.to_numpy()
    test = np.round(test, 6)

    x_test = test[:, 1:]
    y_test = test[:, 0]
    dtest = xgb.DMatrix(x_test, label=y_test)

    for method in ["robust_centergreedy"]:
        eps = eps_vals[dataset].get(method, 0.0)
        for nt in [10, 20, 30, 40]:
            for d in [4, 5, 6, 7, 8, 9, 10]:
                model_path = os.path.join(model_dir, "additional_modelsGBDT_{}_{}_eps{}nt{}d{}.bin".format(
                    dataset, method, eps, nt, d
                ))

                model = xgb.Booster()
                model.load_model(model_path)
                pred_test = model.predict(dtest)
                y_pred_test = [1 if p > 0.5 else 0 for p in pred_test]

                test_acc, test_fpr = eval(y_test, y_pred_test)

                log_text = "{},{},{},{},{},{},{},{}\n".format(
                            "GBDT", dataset, method, eps, nt, d, test_acc, test_fpr
                        )
                log_file.write(log_text)
log_file.close()