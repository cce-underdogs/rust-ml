import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y_names = iris.target_names[iris.target]  

label_map = {"setosa": 0, "versicolor": 1, "virginica": 2}
labels = [label_map[name] for name in y_names]

df = pd.DataFrame(X, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
df["label"] = labels

df.to_csv("iris.csv", index=False)
print("已儲存為 iris.csv，label: setosa→0, versicolor→1, virginica→2")
