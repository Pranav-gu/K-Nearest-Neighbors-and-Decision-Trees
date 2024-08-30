# SMAI Assignment 1
## Name - Pranav Gupta
## Roll No. - 2021101095

All Data Visualisation is done using Matplotlib Library in the Assignment.

In Task 2.2.1, the Data Points are plotted in a scattered manner and not in any histogram, bar chart, etc.
In Task 2.3.1, the KNN Class is made according to the Instructions given.
In Task 2.4.1, the KNN Class Triplets portion is implemented according to the Instructions given.
In Task 2.5.1, the Bash Script has to be run in the following manner: ./eval.sh {absolute path of file}. In case number of arguments passed to the Script is not equal to 1, error is thrown, Proper Error Handling is also done wherever required. It executes a Python File in the Background and prints the required metrics on the Terminal. User is required to run the Values of k and encoder metric to be used for KNN Class. 
In Task 2.6.1, Initial Code is already Vectorised. Also, most optimised KNN Model is taken to be the set of Hyperparameters (Triplet) whose execution time was the lowest out of all possible set of hyperparameters. Initial KNN Model is taken to be some Random Values of K, encoder and Metric while Best KNN Triplet is taken as the Triplet giving maximum acccuracy. Default sklearn is taking the most time out of all the 4 Models (Plots Attached). 

In Task 2.6 (c) inference time vs Train Dataset Size has been interpreted in this way that the Dataset in divided into 2 parts, Test Dataset (whose size is always equal to 0.2 times size of dataset) and Train Dataset whose size is varied from 0.1 times size of Dataset to 0.8 times size of Dataset. In this case, it is also assumed that my Initial KNN Model was the same as Vectorised KNN Model (which whould be the Modt Optimised Model) and so Initial KNN Time is assumed to be same as Optimised KNN Time. Default KNN Time was calculated by making the Call to Library Inbuilt Class Function. The Corresponding Plot is attached in the Jupyter Notebook for reference. Best KNN Model is the KNN Model based upon the Accuracy of the Model according to the Split in that iteration.

In Task 3.1, the Dataset was studied and number of features as well as unique values taken by labels was observed. Also, Box Plot and Bar Charts have also been made for Gender and Income Vs Education Density.

In Task 3.2 and 3.3, Decision Tree Classes were made for 2 Scenarios, Powerset and Multioutput Classification. The I/O is taken to be strictly in the same format as mentioned in the Assignment.
In Task 3.4, Accuracy, F1-Score, Precision, Recall were evaluated only for Macro Case and for Class-Specific Micro-Case as well. 
In Multioutput Classification, the bit set of the Binary Vector means that the Particular Label assigned to that bit is there in the Set whereas in the case of Powerset Classification, we are determining where the bit value of the Required Label of the Test Data Point is 1 so that we know which labels have been predicted by the Model (by looking at the Labels set associated with that particular bit which is set, i.e which labels are a part of in that particular set).
In the Micro-Case, individual Accuracies would be higher because if the Predicted Value and the Evaluated Value differ in only few positions, micro score would take into account the few bits that were set and evaluate the accuracy accordingly, while in the case of Macro Scores, even if 1 bit does not match, it means that value is not correctly predicted by the Classifier and hence is not added to the Accuracy Metric. This is the Reason why such low values of Accuracies are observed in case of Decision Trees.
The evaluation of Micro-Scores have to be done for each class and therefore, detailed calculations are not done separately for each class.
Throughout the Code, References have been mentioned, from where the Code Snippet was referred/taken.