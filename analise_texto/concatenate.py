import pandas as pd

# Load the CSV file
df = pd.read_csv('saidarefor.csv', sep=';', header=None, names=['comments'], engine='python')

# Split the "comments" column into multiple rows
df = df.assign(comments=df['comments'].str.split(',')).explode('comments')

# Reset the index of the DataFrame
df = df.reset_index(drop=True)

# Save the modified DataFrame to a new CSV file
df.to_csv('new_file.csv', index=False)
