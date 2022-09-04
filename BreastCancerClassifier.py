from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()

print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

print(len(training_data))
print(len(training_labels))

classifier= KNeighborsClassifier(n_neighbors = 3)

classifier.fit(training_data, training_labels)

print(classifier.score(validation_data, validation_labels))

k_list = []
accuracies = []

for i in range(1,101):
  classifier2= KNeighborsClassifier(n_neighbors = i)
  classifier2.fit(training_data, training_labels)
  #   print(classifier2.score(validation_data, validation_labels))
  k_list.append(i)
  accuracies.append(classifier2.score(validation_data, validation_labels))

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()