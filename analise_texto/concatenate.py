import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('dataset.csv')
    comment_cols = [col for col in df.columns if col.startswith('comments')]
    comments_df = df.melt(value_vars=comment_cols, var_name='comment_num', value_name='comment')
    comments_df = comments_df.dropna()
    comments_df = comments_df.reset_index(drop=True)

    # Concatenate comments by index
    print("Concatenating comments by index...")
    comments_df = comments_df.groupby(comments_df['comment_num'].astype(str).str.split('/').str[0]).apply(lambda x: ','.join(x['comment']))

    # Create new DataFrame with just one column 'comments'
    print("Creating new DataFrame with just one column 'comments'...")
    comments_df = pd.DataFrame({'comments': comments_df})

    # Write to CSV file
    print("Writing to CSV file...")
    comments_df.to_csv('output.csv', index=False)
    
    print("Done!")