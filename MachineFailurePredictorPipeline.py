import pandas as pd
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('ai4i2020.csv')

class_count_0, class_count_1 = df['Machine failure'].value_counts()

# Separate class
class_0 = df[df['Machine failure'] == 0]
class_1 = df[df['Machine failure'] == 1]
# print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)

class_0_under = class_0.sample(class_count_1)
new_data = pd.concat([class_0_under, class_1], axis=0)

# pre-processed dataset now new_data
print("total class of 1 and 0:")
print(new_data['Machine failure'].value_counts())

# take the letters out of the product ID
new_data['Product ID'] = new_data['Product ID'].str[1:]
# hot encode the Type (H, M, L)
dummies = pd.get_dummies(new_data['Type'])
# drop Type column and attach dummies to new dataset, now called result
new_data = new_data.loc[:, df.columns != 'Type']
result = pd.concat([new_data, dummies], axis=1)

# y: Machine failure
# X: everything else
y = result['Machine failure']
X = result.loc[:, result.columns != 'Machine failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')
print(scores)

