from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np

curr_dir = os.path.dirname(__file__)
fig_name = os.path.join(curr_dir, "test_fig.png")

actual = ["a", "a", "a", "b", "b", "c", "c", "b", "b", "b", "c", "c"]
predicted = ["a", "b", "a", "c", "a", "b", "c", "b", "b", "b", "a", "a"]

cm = metrics.confusion_matrix(actual, predicted, labels=["a", "b", "c"])

print(cm)

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["a", "b", "c"]
)

cm_display.plot()
plt.savefig(fig_name)

per_class_accuracies = {}

for idx, cls in enumerate(["a", "b", "c"]):
    true_negatives = np.sum(np.delete(np.delete(cm, idx, axis=0), idx, axis=1))
    true_negatives = true_negatives

    true_positives = cm[idx, idx]

    per_class_accuracies[cls] = (true_positives + true_negatives) / np.sum(cm)

print("accuracy", per_class_accuracies)

precision = metrics.precision_score(
    actual, predicted, labels=["a", "b", "c"], average=None
)
labels = ["a", "b", "c"]
print("precision", precision)
# print("precision", {label: value for label, value in zip(labels, precision)})

recall = metrics.recall_score(actual, predicted, labels=["a", "b", "c"], average=None)
labels = ["a", "b", "c"]
print("recall", recall)
# print("recall", {label: value for label, value in zip(labels, recall)})

f1_score = metrics.f1_score(actual, predicted, labels=["a", "b", "c"], average=None)
labels = ["a", "b", "c"]
print("f1-score", f1_score)
# print("f1-score", {label: value for label, value in zip(labels, f1_score)})

classification_report = metrics.classification_report(
    actual, predicted, labels=["a", "b", "c"]
)
print(type(classification_report))  # Output: <class 'str'>
print(classification_report)
