"""
eda_practice
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


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
    plt.show()
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

def preprocess(df):
    
    # 利用map取代值
    df['Sex'] = df["Sex"].map({'male':0, "female":1})
    
    # inplace = True 代表取代原有的內容 不新增值
    df['Age'].fillna(df['Age'].median(), inplace = True)
    
    # Binnin pd.cut是利用值做區分位 pd.qcut則是利用分位來區分
    df['AgeGroup'] = pd.cut(
        df['Age'],
        bins=[0, 12, 20, 35, 50, 65, 100],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior", "Elderly"]
    )
    
    # 思路 因為Cabin缺值太多且規則不同 因此改為有無此數值 保留一點關聯但是仍然處理掉缺值問題
    # 也可以改成 只留下最開頭的英文字母 之類的做法
    # notnull()會檢驗格子是不是空值 回傳bool
    # astype則會把它轉成特定值
    df['HasCabin'] = df['Cabin'].notnull().astype(int)
    df.drop('Cabin', axis=1, inplace=True)
    
    # mode是眾數
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    # # 數值化
    # df['Embarked'] = df['Embarked'].map({"S": 0, "C": 1, "Q":2})
    # One-hot
    df['Embarked_S'] = (df['Embarked']=="S").astype(int)
    df['Embarked_C'] = (df['Embarked']=="C").astype(int)
    df['Embarked_Q'] = (df['Embarked']=="Q").astype(int)
    
    # median中位數
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # expand = True代表回傳整個df(會有title和index) False只回傳(series)
    # df[[]]的概念是一次選取多行 得到也是一個df
    df[['Lastname', 'Title']] = df['Name'].str.extract('([A-Za-z]+),\s+([A-Za-z]+)\.', expand=True)
    
    
    # 簡化 title 類別
    title_map = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Lady': 'Rare',
        'Countess': 'Rare', 'Jonkheer': 'Rare', 'Don': 'Rare', 'Sir': 'Rare',
        'Capt': 'Rare'
    }
    
    df['Title'] = df['Title'].map(title_map)
    df['Title'] = df['Title'].fillna('Rare')
    
    df.drop(['Name','Ticket'], axis = 1, inplace=True)
    
    # astype會將儲存格的內容物轉乘category格式 
    # 接著cat.codes會產生對照表 最後只儲存數字
    # 如果要翻回原文 則可以使用cat.categories
    df['Title'] = df['Title'].astype('category').cat.codes
    df['AgeGroup'] = df['AgeGroup'].astype('category').cat.codes
    
    return df    
    

if __name__ == "__main__":
    current_dir = Path(__file__).parent
    print(current_dir)
    file_path = current_dir / "data" / "train.csv"
    df = pd.read_csv(file_path)
    # basic_analysis(df)
    # survived_analysis(df, "Pclass")
    processed_df = preprocess(df)
    print(processed_df.head(5))