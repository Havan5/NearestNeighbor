#Name: Havan Patel
#Date: 1/29/2022
print("Havan Patel")

import numpy as np

# find the appropriate classified lable after computing the nearest neightbor
def findClassifiedInstancesForTestingData(iris_testing_data, iris_training_data, iris_training_label):
    # vector calculation for shortest neighbor
    distances = np.sqrt((np.square(iris_testing_data[:,np.newaxis]-iris_training_data)).sum(axis=2))
    # indexes of min distances
    min_distance_idx = distances.argmin(axis=1)
    # get the training label from the min index
    predicted = [iris_training_label[i] for i in min_distance_idx]
    return predicted

# find the accuracy of the dataset
def getAccuracy(predicted_lable, iris_testing_label):
    print('#, True, Predicted')
    # compute the accuaracy between predicted and testing label.
    count = [printResults(i, iris_testing_label[i], predicted_lable[i]) for i in range(len(iris_testing_label))]
    # print the results
    print(f"Accuracy:{(sum(count)/len(iris_testing_label))*100 : .2f}%")

# print the results
# Params:
#   currentIndex: refers to currentIndex so we can increase the current number
#   testing_lable: current label from the list 
#   predicted_lable: current label from the list
# Returns the classified value
def printResults(currentIndex, testing_label, predicted_lable):
    classifiedInstances = 0
    print(currentIndex + 1, ',', testing_label, ",", predicted_lable)
    # if both predicted label and test label match increase the true/classified vlaue
    if(predicted_lable == testing_label):
        classifiedInstances += 1
    return classifiedInstances 

# read data into vector array?
iris_testing_data = np.loadtxt("iris-testing-data.csv", dtype=float, delimiter=',', usecols=(0, 1, 2, 3), ndmin=2)
iris_training_data = np.loadtxt("iris-training-data.csv", dtype=float, delimiter=',', usecols=(0, 1, 2, 3), ndmin=2)
iris_testing_label = np.loadtxt("iris-testing-data.csv", dtype='str', delimiter=',', usecols=(4))
iris_training_label = np.loadtxt("iris-training-data.csv", dtype='str',  delimiter=',', usecols=(4))

# calculate the clssified instances from testing data set to training data set to find min value/index
predicted = findClassifiedInstancesForTestingData(iris_testing_data, iris_training_data, iris_training_label)

# print the accuracy/results
print()
getAccuracy(predicted, iris_testing_label)

