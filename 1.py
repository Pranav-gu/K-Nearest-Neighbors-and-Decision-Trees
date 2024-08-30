import numpy as np
import heapq
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from prettytable import PrettyTable
import sys


class KNN():
    def __init__(self, k, distance_metric):
        self.k = k
        self.distance_metric = distance_metric

    def euclidean_dist(encoder_point, dataset):
        sum = np.sum((encoder_point-dataset)**2)
        return np.sqrt(sum)

    def manhattan_dist(encoder_point, dataset):
        return np.sum(abs(encoder_point-dataset))
    
    def cosine_dist(encoder_point, dataset):
        return 1-np.dot(encoder_point, dataset) / (np.linalg.norm(encoder_point)*np.linalg.norm(dataset))

    def fit(self, encoder, labels, dataset):
        dist = []
        count = 0
        for i in encoder:
            if (self.distance_metric == 1):
                dist.append((KNN.euclidean_dist(i, dataset), count))
            elif (self.distance_metric == 2):
                dist.append((KNN.manhattan_dist(i, dataset), count))
            elif (self.distance_metric == 3):
                dist.append((KNN.cosine_dist(i, dataset), count))
            count += 1

        smallest = heapq.nsmallest(self.k, dist)
        labels_store = {}
        for i, index in smallest:
            if labels[index] not in labels_store:
                labels_store[labels[index]] = 1
            else:
                labels_store[labels[index]] += 1
                
        max_element = -1
        final_label = ''

        for i in labels_store:
            if (max_element < labels_store[i]):
                max_element = labels_store[i]
                final_label = i

        return final_label


    def performance(self, y_val, y_pred):
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division = 0)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division = 0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division = 0)
        return f1, accuracy, precision, recall
    

# part (2.5)

def task(path, k, metric):
    data = ''
    try:
        data = np.load(path, allow_pickle = True)
    except Exception as e:
        print(e)
        print("Error Reading File. Exiting...")
        exit(0)

    data_res = []
    data_vit = []
    labels = []

    for i in range(len(data)):
        data1 = np.array(data[i][1][0])
        data_res.append(data1)
        data2 = np.array(data[i][2][0])
        data_vit.append(data2)
        labels.append(data[i][3])

    X_train_res, X_test_res, Y_train_res, Y_test_res = train_test_split(data_res, labels, test_size=0.2, random_state=42)

    if (k >= len(data_res)):
        print("K value >= length of dataset. Exiting...")
        exit(0)
    
    if (k == 0):
        print("Please Enter a Valid Value of k.")
        exit(0)
        
    if (metric <= 0 or metric > 3):
        print("Unacceptable value of Metric. Enter Value between 1 to 3 (Inclusive). Exiting...")
        exit(0)

    knn = KNN(k, metric)
    prediction = []
    predicted_val = ''
    for j in range(len(X_test_res)):
        dataset = X_test_res[j]
        predicted_val = knn.fit(X_train_res, Y_train_res, dataset)
        prediction.append(predicted_val)
    f, a, p, r = knn.performance(Y_test_res, prediction)
    
    myTable = PrettyTable(["Accuracy", "F1-Score", "Recall", "Precision"])
    myTable.add_row([a, f, r, p])
    print(myTable)



if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Too Few Arguments")
    
    function_name = sys.argv[1]
    path = sys.argv[2]
    k = int(sys.argv[3])
    metric = int(sys.argv[4])
    if (function_name == 'task'):
        task(path, k, metric)
    else:
        print("Invalid Function Called. Exiting ....")
        exit(0)