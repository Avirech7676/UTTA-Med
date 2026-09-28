# Kaggle notebook cheat-sheet (free GPU)

New notebook → Accelerator: GPU T4/P100 → Internet: On

```python
# cell 1
!pip -q install datasets pyarrow scikit-learn
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
```

Upload `camelyon17_resnet18_source_s42.pt` as a Kaggle dataset and add it to the notebook.
It usually mounts at `/kaggle/input/<your-dataset-name>/`.

Then paste the cells from `notebooks/02_colab_tent_eata.ipynb`, with:

```python
CKPT = Path('/kaggle/input/YOUR-DATASET-NAME/camelyon17_resnet18_source_s42.pt')
SAVE_DIR = Path('/kaggle/working')
```

Download the JSON from `/kaggle/working` when finished.
