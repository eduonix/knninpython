import csv
import random
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#load the dataset and split it into training_set and test_set
def load_dataset(filename, split, training_set_points=[] , test_set=[], training_set_labels=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            training_set_points.append(dataset[x][0:4])
	            training_set_labels.append(dataset[x][-1])
	        else:
	            test_set.append(dataset[x])


#finding accuracy of the programmed knn model
def fetch_accuracy(test_set, knn_predictions):
	correct_prediction = 0
	for x in range(len(test_set)):
		if test_set[x][-1] == knn_predictions[x]:
			correct_prediction += 1
	return (correct_prediction/float(len(test_set))) * 100.0


def main():
	# preparing data
	training_set_points=[]
	training_set_labels=[]
	test_set=[]
	split = 0.67
	load_dataset('flowers_dataset.data', split, training_set_points, test_set, training_set_labels)

	# generating knn_predictions
	training_set_points_np = np.array(training_set_points)
	training_set_labels_np = np.array(training_set_labels)
	knn_predictions=[]
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(training_set_points_np, training_set_labels_np)
	for x in range(len(test_set)):
		test_np=[]
		test_np.append(np.array(test_set[x][0:4]))
		result = neigh.predict(test_np)
		knn_predictions.append(result[0])

	knn_accuracy = fetch_accuracy(test_set, knn_predictions)
	print('Accuracy: ' + repr(knn_accuracy) + '%')
	
main()