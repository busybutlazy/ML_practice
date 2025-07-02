"""
eda_practice
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def basic_analysis(df: pd.DataFrame) -> None:
    """
    basic analysis recommand by Chatgpt.
    """
    print(f"Basic info:\n{df.info()}\n")

    print(f"Head:\n{df.head()}\n")

    print(f"describe:\n{df.describe()}\n")

    print(f"Sum of Missing value:\n{df.isnull().sum()}\n")

    plt.figure(figsize=(10, 5))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.savefig("missing_heatmap.png")
    plt.close()

    print(f"sex distribution:\n{df['Sex'].value_counts()}\n")

    print(f"Embarked distribution:\
          \n{df['Embarked'].value_counts(dropna=False)}")


def survived_analysis(df: pd.DataFrame, column: str) -> None:
    describe_df1 = df[df['Survived'] == 0][column].describe()
    describe_df2 = df[df['Survived'] == 1][column].describe()
    concated_df = pd.concat([describe_df1, describe_df2], axis=1)
    print(f"Now analysis {column}")
    concated_df.columns = [f"{column} of Dead", f"{column} of Alive"]
    print(concated_df)


if __name__ == "__main__":
    df = pd.read_csv(
        "/home/busybutlazy/kaggle_projects/ML_practice/titanic/train.csv")
    # basic_analysis(df)
    survived_analysis(df, "Pclass")
