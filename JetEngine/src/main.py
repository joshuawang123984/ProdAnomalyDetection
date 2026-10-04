import pandas as pd
from helper import add_column_header

def main():
    df = add_column_header("CMAPSSDATA/train_FD001.txt")

    print(df.head())
    print(df.shape)
    print(df.describe().T)
    print(df.isna().sum())
    print(df.nunique())

if __name__ == '__main__':
    main()