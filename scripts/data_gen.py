"""
Data Generation Script
======================
Processes and merges California DMV license plate application data
from multiple source files into a clean, unified dataset.

This script:
1. Loads applications from multiple years (2015 and 2017 data)
2. Cleans invalid/placeholder entries
3. Filters by review reason codes and status
4. Creates appropriate reviewer comments based on approval status
5. Exports to a clean CSV format

Output: data/cali_v2.csv

Usage:
    python data_gen.py

Author: DSF Project Team
Last Updated: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIRS = PROJECT_ROOT / "data_dirs"
OUTPUT_DIR = PROJECT_ROOT / "data"


def load_source_data() -> pd.DataFrame:
    """
    Load and merge California DMV application data from multiple files.
    
    Returns:
        pd.DataFrame: Merged DataFrame from all source files.
    """
    # Load both CSV files
    df1 = pd.read_csv(DATA_DIRS / "ca-license-plates" / "applications.csv")
    df2 = pd.read_csv(DATA_DIRS / "ca-license-plates" / "applications2017.csv")
    
    # Merge the DataFrames
    data = pd.concat([df1, df2], ignore_index=True)
    
    print(f"Loaded {len(df1)} + {len(df2)} = {len(data)} total records")
    return data


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the license plate data.
    
    Performs:
    - Replaces placeholder values with NaN
    - Converts review codes to numeric
    - Drops rows with missing required fields
    
    Args:
        data: Raw DataFrame from source files.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Values that indicate missing/unavailable data
    placeholder_values = ["NO MICRO AVAILABLE", "QUICKWEB UNAVAILABLE", "NO MICRO"]
    
    # Replace placeholder values with NaN
    data['reviewer_comments'] = data['reviewer_comments'].replace(
        placeholder_values, np.nan
    )
    data['customer_meaning'] = data['customer_meaning'].replace(
        placeholder_values, np.nan
    )
    
    # Convert review_reason_code to numeric
    data['review_reason_code'] = pd.to_numeric(
        data['review_reason_code'], 
        errors='coerce'
    )
    
    # Drop rows with missing required fields
    required_columns = ['customer_meaning', 'reviewer_comments', 'plate']
    data = data.dropna(subset=required_columns, how='any')
    
    print(f"After cleaning: {len(data)} records")
    return data


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to relevant review codes and statuses.
    
    Review reason codes 1-7 represent different categories of review:
    1 = Duplicate/Similar plate
    2 = Offensive content
    3 = Gang-related
    4 = Drug-related
    5 = Sexual content
    6 = Profanity
    7 = Other
    
    Args:
        data: Cleaned DataFrame.
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Filter by review reason code (1-7 are the relevant categories)
    data = data[data['review_reason_code'].between(1, 7)]
    
    # Keep only approved (Y) or rejected (N) applications
    data = data[data['status'].isin(['Y', 'N'])]
    
    print(f"After filtering: {len(data)} records")
    return data


def create_target_comments(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create appropriate target comments based on application status.
    
    For rejected plates (N): Use reviewer's comments (explains the issue)
    For approved plates (Y): Use customer's meaning (what they intended)
    
    Args:
        data: Filtered DataFrame.
        
    Returns:
        pd.DataFrame with unified reviewer_comments column.
    """
    data = data.copy()
    
    # Update reviewer_comments based on status
    data['reviewer_comments'] = np.where(
        data['status'] == 'N',
        data['reviewer_comments'],   # Rejected: use reviewer comment
        data['customer_meaning']      # Approved: use customer meaning
    )
    
    return data


def finalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Finalize data types and select output columns.
    
    Args:
        data: Processed DataFrame.
        
    Returns:
        pd.DataFrame: Final DataFrame ready for export.
    """
    data = data.copy()
    
    # Ensure string types
    data["plate"] = data["plate"].astype(str)
    data["reviewer_comments"] = data["reviewer_comments"].astype(str)
    
    return data


def main():
    """Main data generation pipeline."""
    print("=" * 60)
    print("California DMV Data Processing Pipeline")
    print("=" * 60)
    
    # Load source data
    print("\n1. Loading source data...")
    data = load_source_data()
    
    # Clean data
    print("\n2. Cleaning data...")
    data = clean_data(data)
    
    # Filter data
    print("\n3. Filtering by review codes and status...")
    data = filter_data(data)
    
    # Create target comments
    print("\n4. Creating target comments...")
    data = create_target_comments(data)
    
    # Finalize
    print("\n5. Finalizing data...")
    data = finalize_data(data)
    
    # Save output
    output_path = OUTPUT_DIR / "cali_v2.csv"
    data.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print(f"✓ Data saved to: {output_path}")
    print(f"  Total records: {len(data)}")
    print(f"  Columns: {list(data.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
