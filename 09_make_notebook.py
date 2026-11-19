"""09_make_notebook.py - build results.ipynb and execute it, so it opens in VS Code with all outputs shown.

Usage: python 09_make_notebook.py
Output: results.ipynb (already executed)
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from common import ROOT

CELLS = [
    ("md", "# Semi-supervised OCT classification: ITS and ITS+\n\n"
           "Dataset: NEH_UT_2021 (Noor Eye Hospital), classes NORMAL / DRUSEN / CNV.\n\n"
           "- **Supervised**: EfficientNet-B0 trained on the labelled scans only.\n"
           "- **ITS (Paper 1)**: iterative teacher-student pseudo-labelling as published by Alizadeh et al.\n"
           "- **ITS+ (ours)**: the same loop with a two-head network, class-adaptive full / partial / reject "
           "admission, a neighbouring-scan agreement check, strong augmentation with a ramped pseudo-label "
           "loss, and a final fine-tune (taken from Papers 2 and 3).\n\n"
           "Every cell below was executed; the outputs are stored in this file."),
    ("md", "## 1. Live check: load the saved models and classify the test set"),
    ("code", "%run demo.py --fraction 0.10 --seed 0"),
    ("md", "## 2. All test results"),
    ("code", "import pandas as pd\n"
             "pd.set_option('display.width', 200)\n"
             "res = pd.read_csv('outputs/results.csv')\n"
             "cols = ['label', 'fraction', 'seed', 'best_round', 'accuracy', 'macro_f1', 'macro_auc', "
             "'under_grading_rate']\n"
             "res[cols].round(4)"),
    ("md", "## 3. Figures"),
    ("code", "from IPython.display import Image, display\n"
             "for name in ['08_comparison.png', '08_confusions.png', '08_rounds.png', '03_augmentations.png']:\n"
             "    print(name)\n"
             "    display(Image(filename=f'outputs/{name}'))"),
    ("md", "## 4. Teacher-student rounds (seed 0, 10% labels)\n\n"
           "Pseudo-label precision is measured against the hidden true labels for analysis only."),
    ("code", "cols = ['round', 'tau', 'kept', 'val_macro_f1', 'val_accuracy', 'admitted', 'coverage', "
             "'full_precision', 'n_PARTIAL', 'precision_PARTIAL']\n"
             "for run in ['its_f010_s0', 'its_plus_f010_s0']:\n"
             "    print(run)\n"
             "    display(pd.read_csv(f'outputs/runs/{run}/rounds.csv').reindex(columns=cols).round(4))"),
    ("md", "## 5. Training log of ITS+ (last 40 lines)"),
    ("code", "print(''.join(open('outputs/runs/its_plus_f010_s0/log.txt').readlines()[-40:]))"),
]


def main():
    nb = new_notebook(cells=[new_markdown_cell(s) if k == "md" else new_code_cell(s) for k, s in CELLS])
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT)}})
    nbformat.write(nb, ROOT / "results.ipynb")
    print("saved results.ipynb (executed, outputs embedded)")


if __name__ == "__main__":
    main()
