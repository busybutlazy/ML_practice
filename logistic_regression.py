import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from eda_practice import preprocess


current_dir = Path(__file__).parent

data_path = current_dir/"data"/"train.csv"
df = pd.read_csv(data_path)
process_df = preprocess(df)


X = df.drop(["Survived", "PassengerId"], axis=1)
y = df["Survived"]

X_train, X_val, y_train, y_valid = train_test_split(X,y,test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train , y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_valid, y_pred)
print(f"accuracy:{accuracy}")

cm = confusion_matrix(y_pred, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Survived", "Survived"])
disp.plot()
