import pandas as pd
import numpy as np

# Load both CSV files into DataFrames
df1 = pd.read_csv('../data_dirs/ca-license-plates/applications.csv')
df2 = pd.read_csv('../data_dirs/ca-license-plates/applications2017.csv')

# Merge the DataFrames on their column headers (i.e., column names)
data = pd.concat([df1, df2], ignore_index=True)

# Replace specific values with NaN in 'reviewer_comments' and 'customer_meaning'
data['reviewer_comments'] = data['reviewer_comments'].replace(
    ["NO MICRO AVAILABLE", "QUICKWEB UNAVAILABLE", "NO MICRO"], np.nan
)
data['customer_meaning'] = data['customer_meaning'].replace(
    ["NO MICRO AVAILABLE", "QUICKWEB UNAVAILABLE", "NO MICRO"], np.nan
)

# Convert 'review_reason_code' to numeric, replacing non-numeric values with NaN
data['review_reason_code'] = pd.to_numeric(data['review_reason_code'], errors='coerce')

# Drop columns where all values in 'customer_meaning' are NaN
data = data.dropna(subset=['customer_meaning', 'reviewer_comments', 'plate'], how='any')

# Filter rows with 'review_reason_code' between 1 and 7
filtered_data = data[data['review_reason_code'].between(1, 7)]

# Keep rows with 'status' as 'Y' or 'N'
filtered_data = filtered_data[filtered_data['status'].isin(['Y', 'N'])]

# Update 'reviewer_comments' based on the 'status' column
filtered_data['reviewer_comments'] = np.where(
    filtered_data['status'] == 'N',
    filtered_data['reviewer_comments'],
    filtered_data['customer_meaning']
)


# Convert 'plate' and 'reviewer_comments' to strings
filtered_data["plate"] = filtered_data["plate"].astype(str)
filtered_data["reviewer_comments"] = filtered_data["reviewer_comments"].astype(str)

# Save the filtered DataFrame to a new CSV file
filtered_data.to_csv('../data/cali_v2.csv', index=False)
print("File 'cali.csv' has been saved successfully!")
