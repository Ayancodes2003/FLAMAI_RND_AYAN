"""
Smoke tests for environment and dataset accessibility.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import src


def test_package_import():
    """Verify src package can be imported and has version."""
    assert hasattr(src, "__version__")
    assert src.__version__ == "0.1.0"


def test_dataset_exists_and_valid():
    """Verify xy_data.csv is present, non-empty, and has expected columns."""
    dataset_path = Path("xy_data.csv")
    assert dataset_path.exists(), "xy_data.csv should exist in repository root"
    
    df = pd.read_csv(dataset_path)
    assert df.shape == (1500, 2), f"Expected shape (1500, 2), got {df.shape}"
    assert list(df.columns) == ["x", "y"], f"Expected columns ['x', 'y'], got {list(df.columns)}"
    assert df.isnull().sum().sum() == 0, "Dataset should have no missing values"
    assert not df.empty, "Dataset should not be empty"
