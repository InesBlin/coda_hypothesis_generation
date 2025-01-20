# -*- coding: utf-8 -*-
"""
Analyse results from complex classification systems
"""
import os
import pandas as pd

FOLDER = "./experiments/classification/complex"

def print_overleaf_row(res):
    order = ["regular", "study_mod", "var_mod"]
    line = []
    for o in order:
        line += res.get(o, ["", "", ""])
    print(" & ".join(line) + " \\\\")

def main():
    models = os.listdir(FOLDER)
    models = [x for x in models if os.path.isdir(os.path.join(FOLDER, x))]
    for m in models:
        print(m)
        curr_folder = os.path.join(FOLDER, m)
        metrics_files = [x for x in os.listdir(curr_folder) if x.endswith("_metrics.csv")]
        res = {}
        for mf in metrics_files:
            fp = os.path.join(curr_folder, mf)
            df = pd.read_csv(fp)
            print(f" {mf} | {df.shape[0]}")
            row = df.sort_values(by="acc_val", ascending=False).iloc[0]
            print("acc_train: ", round(row.acc_train, 2), " | acc_val: ", round(row.acc_val, 2))
            res[mf.replace("h_", "").replace("_es_d_metrics.csv", "")] = [str(df.shape[0]), str(round(row.acc_train, 2)), str(round(row.acc_val, 2))]
        print_overleaf_row(res=res)
        print("===")


if __name__ == '__main__':
    main()