import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

file_path = 'C:/Users/Jose/Downloads/archive/cleaned_merged_heart_dataset.csv'
data = pd.read_csv(file_path)

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


decision_tree_id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
decision_tree_id3.fit(X_train, y_train)
y_pred_entropy = decision_tree_id3.predict(X_test)
accuracy_id3 = accuracy_score(y_test, y_pred_entropy)


decision_tree_gini = DecisionTreeClassifier(criterion='gini', random_state=42)
decision_tree_gini.fit(X_train, y_train)
y_pred_gini = decision_tree_gini.predict(X_test)
accuracy_gini = accuracy_score(y_test, y_pred_gini)



print(accuracy_id3*100,accuracy_gini*100)
