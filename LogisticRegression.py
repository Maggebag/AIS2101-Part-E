from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from data_utils import import_and_split_data

X_train_scaled, X_test_scaled, y_train, y_test = import_and_split_data('dataset/cleaned_dataset.csv')

print(X_train_scaled.shape, X_test_scaled.shape)

test1 = LogisticRegression(random_state=0,
                           max_iter=1000).fit(X_train_scaled, y_train)

test1_y_predict = test1.predict(X_test_scaled)
test1_train_score = test1.score(X_train_scaled, y_train)
test1_accuracy = accuracy_score(y_test, test1_y_predict)
test1_classification_report = classification_report(y_test, test1_y_predict)
roc_auc_test1 = roc_auc_score(y_test, test1.predict_proba(X_test_scaled)[:, 1])

test2 = LogisticRegression(random_state=0,
                           C=20,
                           class_weight='balanced',
                           penalty='elasticnet',
                           solver='saga',
                           l1_ratio=0.1,
                           max_iter=1000).fit(X_train_scaled, y_train)

test2_y_predict = test2.predict(X_test_scaled)
test2_train_score = test2.score(X_train_scaled, y_train)
test2_accuracy = accuracy_score(y_test, test2_y_predict)
test2_classification_report = classification_report(y_test, test2_y_predict)
roc_auc_test2 = roc_auc_score(y_test, test2.predict_proba(X_test_scaled)[:, 1])


test3 = LogisticRegression(random_state=0,
                           C=0.00001,
                           class_weight='balanced',
                           solver='liblinear',
                           penalty='l2',
                           max_iter=1000).fit(X_train_scaled, y_train)

test3_y_predict = test3.predict(X_test_scaled)
test3_train_score = test3.score(X_train_scaled, y_train)
test3_accuracy = accuracy_score(y_test, test3_y_predict)
test3_classification_report = classification_report(y_test, test3_y_predict)
roc_auc_test3 = roc_auc_score(y_test, test3.predict_proba(X_test_scaled)[:, 1])

# Plots
accuracies = [test1_accuracy, test2_accuracy, test3_accuracy]
max_accuracy = max(accuracies)
min_accuracy = min(accuracies)
difference = max_accuracy - min_accuracy

test_scores = [test1_train_score, test2_train_score, test3_train_score]

max_score = max(test_scores)
min_score = min(test_scores)
difference_test = max_score - min_score

labels = ['Test 1', 'Test 2', 'Test 3']
plt.figure(figsize=(8, 6))
plt.bar(labels, accuracies, color=['blue', 'orange', 'green'])
plt.xlabel('')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Logistic Regression Models')
plt.ylim(min_accuracy-difference, max_accuracy + difference)
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(labels, test_scores, color=['blue', 'orange', 'green'])
plt.xlabel('')
plt.ylabel('Training Score')
plt.title('Training Score Comparison of Logistic Regression Models')
plt.ylim(min_score - difference_test, max_score + difference_test)
plt.show()

# Roc curves from chatgpt
fpr_test1, tpr_test1, _ = roc_curve(y_test, test1.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr_test1, tpr_test1, label='Test 1 (AUC = {:.2f})'.format(roc_auc_test1))

fpr_test2, tpr_test2, _ = roc_curve(y_test, test2.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr_test2, tpr_test2, label='Test 2 (AUC = {:.2f})'.format(roc_auc_test2))

fpr_test3, tpr_test3, _ = roc_curve(y_test, test3.predict_proba(X_test_scaled)[:,1])
plt.plot(fpr_test3, tpr_test3, label='Test 3 (AUC = {:.2f})'.format(roc_auc_test3))

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')

# Set labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)

# Show plot
plt.show()
