# genetic algorithmes and machine learning based feature selection for breast cancer detection
 In this project, a hybrid Genetic algorithmes/SVM method for gene selection is presentd. The proposed method uses GAs to search for 
a subset of genes that optimize the detection of breast cancer. Thus, data are better represented and classification of unknown samples may become easier.
## pipeline
### data preprocessing
cleaning the dataset and performing the necessary preprocessing 
### feature selection
using select k best algorithm to reduce the data's dimentions from 18000 to 1000 features 
### training phase and choosing the best features
SVM classifier was trained on the dataset  and  the svm's prediction score was used as the fitness function in the GA to choose the least number of features that gives the heighst fitness score.
### testing
the resulting optimum set of features which were only 4 genes out of the original 18000 was tested using KNN which was able to detect malignant records with an accuracy of 94% and also the gene's descriptions in NCBI and GeneCards websites confirm that their mutation causes cancer.
