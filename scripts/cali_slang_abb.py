"""
California Plates + Slang Abbreviation Data Merger
=================================================
Combines California vanity plate data with slang/abbreviation data
to create an enhanced training dataset.

This allows models to learn from both:
1. Real vanity plate examples (plate → meaning)
2. General slang/abbreviations (term → expansion)

Sources:
- data/cali.csv: California DMV vanity plate data
- data/slang_abb.csv: Merged slang and abbreviations

Output: data/cali_slang_abb.csv

Usage:
    python cali_slang_abb.py

Author: AMS 560
Last Updated: December 2025
"""

import pandas as pd
from pathlib import Path

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_california_plates(filepath: Path) -> pd.DataFrame:
    """
    Load and extract relevant columns from California plate data.
    
    Args:
        filepath: Path to the California plates CSV file.
        
    Returns:
        pd.DataFrame with 'acronym' and 'expansion' columns.
    """
    df = pd.read_csv(filepath)
    
    # Extract plate and comments, rename to match slang format
    df = df[['plate', 'reviewer_comments']].copy()
    df = df.rename(columns={
        'plate': 'acronym',
        'reviewer_comments': 'expansion'
    })
    
    # Clean data
    df = df.dropna()
    df['acronym'] = df['acronym'].astype(str).str.strip()
    df['expansion'] = df['expansion'].astype(str).str.strip()
    
    print(f"Loaded {len(df)} California plate entries")
    return df


def load_slang_abbreviations(filepath: Path) -> pd.DataFrame:
    """
    Load slang and abbreviation data.
    
    Args:
        filepath: Path to the slang_abb CSV file.
        
    Returns:
        pd.DataFrame with 'acronym' and 'expansion' columns.
    """
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} slang/abbreviation entries")
    return df


def merge_datasets(slang_df: pd.DataFrame, cali_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge slang and California plate datasets.
    
    Args:
        slang_df: Slang/abbreviation DataFrame.
        cali_df: California plates DataFrame.
        
    Returns:
        pd.DataFrame: Combined dataset.
    """
    merged = pd.concat([slang_df, cali_df], ignore_index=True)
    
    # Remove duplicates
    merged = merged.drop_duplicates()
    
    # Remove empty entries
    merged = merged.dropna()
    merged = merged[merged['acronym'].str.strip() != '']
    merged = merged[merged['expansion'].str.strip() != '']
    
    print(f"Merged dataset: {len(merged)} entries")
    return merged


def main():
    """Main data merging pipeline."""
    print("=" * 60)
    print("California Plates + Slang/Abbreviation Merger")
    print("=" * 60)
    
    # Define file paths
    cali_file = DATA_DIR / "cali.csv"
    slang_abb_file = DATA_DIR / "slang_abb.csv"
    output_file = DATA_DIR / "cali_slang_abb.csv"
    
    # Load data
    print("\n1. Loading slang and abbreviations...")
    slang_df = load_slang_abbreviations(slang_abb_file)
    
    print("\n2. Loading California plate data...")
    cali_df = load_california_plates(cali_file)
    
    # Merge datasets
    print("\n3. Merging datasets...")
    merged_df = merge_datasets(slang_df, cali_df)
    
    # Save output
    merged_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print(f"✓ Merged file saved to: {output_file}")
    print(f"  Total entries: {len(merged_df)}")
    print("=" * 60)
    
    # Show statistics
    print("\nDataset composition:")
    print(f"  - Slang/Abbreviations: ~{len(slang_df)} entries")
    print(f"  - California Plates: ~{len(cali_df)} entries")
    
    # Show sample entries
    print("\nSample entries:")
    print(merged_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
