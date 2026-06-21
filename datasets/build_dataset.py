from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from datasets.augmentations import get_tta_transforms

class StreamingMedicalDataset(Dataset):
    """
    Standardized dataset loader for Test-Time Adaptation streaming.
    Outputs the original image alongside its weakly and strongly augmented views.
    """
    def __init__(self, csv_path, input_size=224, root_dir=None):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir) if root_dir else self.csv_path.parent
        self.data_info = pd.read_csv(csv_path)
        required_columns = {"filepath", "label"}
        missing = required_columns.difference(self.data_info.columns)
        if missing:
            raise ValueError(f"Dataset CSV is missing required columns: {sorted(missing)}")
        
        # Load the asymmetric augmentation pipelines
        self.base_tf, self.weak_tf, self.strong_tf = get_tta_transforms(input_size)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        img_path = Path(row['filepath'])
        if not img_path.is_absolute():
            img_path = self.root_dir / img_path
        label = int(row['label'])
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        img = Image.open(img_path).convert('RGB')

        # Generate the multi-view inputs required for consistency regularization
        x_clean = self.base_tf(img)
        x_weak = self.weak_tf(img)
        x_strong = self.strong_tf(img)

        return x_clean, x_weak, x_strong, label, str(img_path)
