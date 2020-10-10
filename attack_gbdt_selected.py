import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score
import os
from xgbKantchelianAttack import main
from sklearn.datasets import load_svmlight_file
import pickle
import os
import numpy as np

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

test_path = {
    "breast_cancer": "../DevRobustTrees/data/breast_cancer_scale0.test",
    "binary_mnist": "../DevRobustTrees/data/binary_mnist0.t",
    "cod-rna": "../DevRobustTrees/data/cod-rna_s.t",
    "ijcnn": "../DevRobustTrees/data/ijcnn1s0.t"
}

model_dir = 'models'
output_dir = 'log'

log_fname = 'GBDT_attack_result_first500.csv'
if os.path.isfile(log_fname):
    log_file = open(log_fname, "a+")
else:
    log_file = open(log_fname, "w")
    log_file.write("type,dataset,method,eps,nt,d,Linf,time\n")

for dataset in ["cod-rna"]:

    for method in ["robust_centergreedy"]:
        eps = eps_vals[dataset].get(method, 0.0)
        for nt in [10, 20, 30, 40]:
            for d in [4, 5, 6, 7, 8, 9, 10]:
                model_path = os.path.join(model_dir, "additional_modelsGBDT_{}_{}_eps{}nt{}d{}.bin".format(
                    dataset, method, eps, nt, d
                ))

                json_path = '%s.json' % model_path.split('.bin')[0]
                cmd = 'python3 save_xgboost_model_to_json.py \
                        --model_path %s \
                        --output %s' % (model_path, json_path)
                os.system(cmd)

                args = dict()
                args["data"] = test_path[dataset]
                args["model"] = model_path
                args["model_type"] = "xgboost"
                args["model_json"] = json_path
                args["num_classes"] = 2
                args["offset"] = 0
                args["order"] = -1  # only attack linf
                args["num_attacks"] = 500
                args["guard_val"] = 1e-6
                args["round_digits"] = 20
                args["round_data"] = 6
                args["weight"] = "No weight"
                args["out"] = os.path.join(output_dir, "{}_{}_eps{}nt{}d{}_order{}.csv".format(
                    dataset, method, eps, nt, d, -1
                ))
                args["adv"] = os.path.join(output_dir, "{}_{}_eps{}nt{}d{}_order{}.pickle".format(
                    dataset, method, eps, nt, d, -1
                ))
                args["threads"] = 8
                if zero_based[dataset]:
                    fstart = 0
                else:
                    fstart = 1
                args["feature_start"] = fstart
                args["initial_check"] = False
                args["no_shuffle"] = False
                avg_l0, avg_l1, avg_l2, avg_linf, avg_time = main(args)

                log_text = "{},{},{},{},{},{},{},{}\n".format(
                            "GBDT", dataset, method, eps, nt, d, avg_linf, avg_time
                        )
                log_file.write(log_text)
log_file.close()
