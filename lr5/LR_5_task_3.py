import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier


def visualize_classifier(classifier, X, y, title=''):
    plt.figure()
    X_min, X_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
    plt.title(title)
    plt.show()


# Завантаження вхідних даних
input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')

# Візуалізація початкових даних для перевірки коректності
plt.figure()
plt.hist(data.flatten(), bins=50)
plt.title('Розподіл початкових даних')
plt.show()

# Перевірка на наявність NaN значень
if np.isnan(data).any():
    print("Дані містять NaN значення. Виконується очищення даних...")
    # Замініть NaN значення на 0 (або інше значення за потреби)
    data = np.nan_to_num(data, nan=0.0, posinf=None, neginf=None)

# Додаткова перевірка на некоректні значення
if np.isinf(data).any():
    print("Дані містять нескінченні значення. Виконується очищення даних...")
    data = np.nan_to_num(data, nan=0.0, posinf=1e308, neginf=-1e308)

X, y = data[:, :-1], data[:, -1]

# Поділ вхідних даних на два класи на підставі міток
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])

# Розбиття даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Визначення сітки значень параметрів
parameter_grid = [
    {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
    {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
]
metrics = ['precision_weighted', 'recall_weighted']

for metric in metrics:
    print("\n #### Searching optimal parameters for", metric)

    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0),
        parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

    print("\nGrid scores for the parameter grid:")
    results = classifier.cv_results_
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print(params, '-->', round(mean_score, 3))
    print("\nBest parameters:", classifier.best_params_)

y_pred = classifier.predict(X_test)
print("\nPerformance report:\n")
print(classification_report(y_test, y_pred))
