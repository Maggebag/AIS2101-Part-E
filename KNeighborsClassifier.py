from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from data_utils import import_and_split_data

X_train_scaled, X_test_scaled, y_train, y_test = import_and_split_data('dataset/cleaned_dataset.csv')

# Test 1 with default parameters
knn_model_1 = KNeighborsClassifier().fit(X_train_scaled, y_train)
knn_pred_1 = knn_model_1.predict(X_test_scaled)
knn_score_1 = knn_model_1.score(X_train_scaled, y_train)
knn_accuracy_1 = accuracy_score(y_test, knn_pred_1)
knn_report_1 = classification_report(y_test, knn_pred_1)
knn_roc_auc_1 = roc_auc_score(y_test, knn_model_1.predict_proba(X_test_scaled)[:, 1])

# Test 2 with modified parameters, more neighbours, uniform weight distrubution and brute force search
knn_model_2 = KNeighborsClassifier(n_neighbors=10,
                                   weights='uniform',
                                   algorithm='brute').fit(X_train_scaled, y_train)
knn_pred_2 = knn_model_2.predict(X_test_scaled)
knn_score_2 = knn_model_2.score(X_train_scaled, y_train)
knn_accuracy_2 = accuracy_score(y_test, knn_pred_2)
knn_report_2 = classification_report(y_test, knn_pred_2)
knn_roc_auc_2 = roc_auc_score(y_test, knn_model_2.predict_proba(X_test_scaled)[:, 1])

# Test 3 with modified parameters, same neighbour count as earlier, distance weight distrubution with KDTree algorithm
knn_model_3 = KNeighborsClassifier(n_neighbors=10,
                                   weights='distance',
                                   algorithm='kd_tree').fit(X_train_scaled, y_train)
knn_pred_3 = knn_model_3.predict(X_test_scaled)
knn_score_3 = knn_model_3.score(X_train_scaled, y_train)
knn_accuracy_3 = accuracy_score(y_test, knn_pred_3)
knn_report_3 = classification_report(y_test, knn_pred_3)
knn_roc_auc_3 = roc_auc_score(y_test, knn_model_3.predict_proba(X_test_scaled)[:, 1])

# Plots
accuracies = [knn_accuracy_1, knn_accuracy_2, knn_accuracy_3]
max_accuracy = max(accuracies)
min_accuracy = min(accuracies)
difference = max_accuracy - min_accuracy

test_scores = [knn_score_1, knn_score_2, knn_score_3]

max_score = max(test_scores)
min_score = min(test_scores)
difference_test = max_score - min_score

labels = ['Test 1', 'Test 2', 'Test 3']
plt.figure(figsize=(8, 6))
plt.bar(labels, accuracies, color=['blue', 'orange', 'green'])
plt.xlabel('')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of KNN models')
plt.ylim(min_accuracy-difference, max_accuracy + difference)
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(labels, test_scores, color=['blue', 'orange', 'green'])
plt.xlabel('')
plt.ylabel('Training Score')
plt.title('Training Score Comparison of KNN models')
plt.ylim(min_score - difference_test, max_score + difference_test)
plt.show()

fpr_knn1, tpr_knn1, _ = roc_curve(y_test, knn_model_1.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr_knn1, tpr_knn1, label='KNN Test 1 (AUC = {:.2f})'.format(knn_roc_auc_1))

fpr_knn2, tpr_knn2, _ = roc_curve(y_test, knn_model_2.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr_knn2, tpr_knn2, label='KNN Test 2 (AUC = {:.2f})'.format(knn_roc_auc_2))

fpr_knn3, tpr_knn3, _ = roc_curve(y_test, knn_model_3.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr_knn3, tpr_knn3, label='KNN Test 3 (AUC = {:.2f})'.format(knn_roc_auc_3))

# Plot random guessing line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for KNN Models')
plt.legend()
plt.grid(True)

# Show plot
plt.show()