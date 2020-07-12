from code import *

args = get_args()
traning_data = args['traning_data']
testing_data = args['testing_data']

training_features, traning_labels = get_data_details(traning_data)

testing_features, testing_labels = get_data_details(testing_data)

print("\n\n=-=-=-=-=-=-=- Decision Tree Classifier -=-=-=-=-=-=-=-\n")

attack_classifier = tree.DecisionTreeClassifier()

attack_classifier = attack_classifier.fit(training_features, traning_labels)

predictions = attack_classifier.predict(testing_features)

print("The precision of the Decision Tree Classifier is: " + str(get_occuracy(testing_labels,predictions, 1)) + "%")
