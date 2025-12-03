"""
Slang and Abbreviation Data Merger
==================================
Combines slang definitions and abbreviation expansions into a single
training dataset for language understanding models.

Sources:
- data_dirs/slang/slang.csv: Slang words and definitions
- data_dirs/slang/abbrevations.csv: Abbreviations and expansions

Output: data/slang_abb.csv

Usage:
    python slang_abb_data.py

Author: DSF Project Team
Last Updated: December 2025
"""

import pandas as pd
from pathlib import Path

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIRS = PROJECT_ROOT / "data_dirs"
OUTPUT_DIR = PROJECT_ROOT / "data"


def load_abbreviations(filepath: Path) -> pd.DataFrame:
    """
    Load abbreviation data from CSV.
    
    Args:
        filepath: Path to the abbreviations CSV file.
        
    Returns:
        pd.DataFrame with 'acronym' and 'expansion' columns.
    """
    # File has no header, add column names
    df = pd.read_csv(filepath, header=None, names=["acronym", "expansion"])
    print(f"Loaded {len(df)} abbreviations")
    return df


def load_slang(filepath: Path) -> pd.DataFrame:
    """
    Load slang data from CSV.
    
    Args:
        filepath: Path to the slang CSV file.
        
    Returns:
        pd.DataFrame with 'acronym' and 'expansion' columns.
    """
    df = pd.read_csv(filepath)
    
    # Drop ID column if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    print(f"Loaded {len(df)} slang terms")
    return df


def merge_datasets(slang_df: pd.DataFrame, abbrev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge slang and abbreviation datasets.
    
    Args:
        slang_df: Slang terms DataFrame.
        abbrev_df: Abbreviations DataFrame.
        
    Returns:
        pd.DataFrame: Combined dataset.
    """
    merged = pd.concat([slang_df, abbrev_df], ignore_index=True)
    
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
    print("Slang and Abbreviation Data Merger")
    print("=" * 60)
    
    # Define file paths
    abbreviation_file = DATA_DIRS / "slang" / "abbrevations.csv"
    slang_file = DATA_DIRS / "slang" / "slang.csv"
    output_file = OUTPUT_DIR / "slang_abb.csv"
    
    # Load data
    print("\n1. Loading abbreviations...")
    abbrev_df = load_abbreviations(abbreviation_file)
    
    print("\n2. Loading slang terms...")
    slang_df = load_slang(slang_file)
    
    # Merge datasets
    print("\n3. Merging datasets...")
    merged_df = merge_datasets(slang_df, abbrev_df)
    
    # Save output
    merged_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print(f"✓ Merged file saved to: {output_file}")
    print(f"  Total entries: {len(merged_df)}")
    print("=" * 60)
    
    # Show sample entries
    print("\nSample entries:")
    print(merged_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
