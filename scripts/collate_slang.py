import pandas as pd

# File paths
abbreviation_file = "../data_dirs/slang/abbrevations.csv"
slang_file = "../data_dirs/slang/slang.csv"
output_file = "../data/slang_abb.csv"

# Read the abbreviation CSV file
abbreviation_df = pd.read_csv(abbreviation_file, header=None, names=["acronym", "expansion"])

# Read the slang CSV file
slang_df = pd.read_csv(slang_file)

# Drop the ID column from slang_df if not needed
slang_df = slang_df.drop(columns=["id"])

# Concatenate the two dataframes
merged_df = pd.concat([slang_df, abbreviation_df], ignore_index=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv(output_file, index=False)

print(f"Merged file saved to {output_file}")
