'''
Iris classification testing. REV.
Following scikit-learn supervised learning documentation.
'''
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
#------------------------------------------------------------------------------
#k nearest neighbor prediction function
#instance based , non-generalising learning. simple majority vote of nearest neighbours
from sklearn.neighbors import KNeighborsClassifier   ##k nearest neighbors

def train_and_predict_knn(X_train, y_train, X_test): 
    knn = KNeighborsClassifier()        #can input nnumber of nearestneighbors here. this can change the outcome
    knn.fit(X_train, y_train)
    return np.array(knn.predict(X_test))

#------------------------------------------------------------------------------
#NU support vector classification
#draws vectors to classify, regress, outlier-detect
#good in high dim spaces. good where n-dim > n-samples. memory efficient. easy to overfit if n-dim > n-samples.
from sklearn.svm import NuSVC                        ##support vector machines

def train_and_predict_SVM(X_train, y_train, X_test):
    nu=NuSVC(kernel='rbf',gamma='auto')    #Kernel can be linear, poly, rbf, sigmoid. - correspons to the lines drawn between regions
    nu.fit(X_train, y_train)
    return nu.predict(X_test)

#------------------------------------------------------------------------------
#Stochastic Gradient Descent
#for fitting linear classifier and regressors under convex loss functions.
#efficient, very tunable.   
from sklearn.linear_model import SGDClassifier       ##stochastic gradient descent

def train_and_predict_SGD(X_train, y_train, X_test):
    SGD= SGDClassifier(loss="perceptron", penalty="l2")
    SGD.fit(X_train,y_train)
    return SGD.predict(X_test)

#SGD can be used to: predict confidence scores of samples, fit learn model with SGD, get params for this estimator, predict class labels (like this e.g.),return mean accuracy on given test and labels data, 

#------------------------------------------------------------------------------
#Bayes theore. Naice assumption of conitional independence between every pair of features
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB           ##Naive Bayes

def train_and_predict_GNB(X_train, y_train, X_test):
    #gnb = GaussianNB()                     #GaussianNB
    #gnb=MultinomialNB()                     #MultinomialNB for multinomially distributed data       
    gnb=ComplementNB()                      #Complement Naive Bayes (MNB for imbalanced data sets). OutperformsMNB on text lasification tasks
    return gnb.fit(X_train, y_train).predict(X_test)
    #gnb.fit(X_train, y_train)
    #return gnb.predict(X_test)
    
#+Bernoulli NB. For data istributed according to MV bernoulli dists
#+categorical NB. categorically dist data.
#+Out of core NB. large scale problems, when full training set might not fit in memory. does partial fit methods, incrememtally fits.

#------------------------------------------------------------------------------
#Decision trees creates model which predicts avlue of a target by using simple rules inferred from data.If then else decision rules. 
#Deeper tree -> more complex decision rules -> more fitted model.
from sklearn import tree                             ##DEcision trees

def train_and_predict_tree(X_train, y_train, X_test):
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    tree.plot_tree(clf) 
    
    return clf.predict(X_test)

#------------------------------------------------------------------------------
#Supervised neural network
from sklearn.neural_network import MLPClassifier

def train_and_predict_sNN(x,y,Y):
    clf = MLPClassifier(solver='sgd',hidden_layer_sizes=(10,10,15), random_state=1, verbose=True)
    #solvers:‘lbfgs’ is an optimizer in the family of quasi-Newton methods.‘sgd’ refers to stochastic gradient descent.‘adam’ works pretty well on relatively large datasets
    clf.fit(x, y)
    return clf.predict(Y)


#------------------------------------------------------------------------------
def main():
        iris = datasets.load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size=0.3)

        y_pred = train_and_predict_sNN(X_train, y_train, X_test)
        if y_pred is not None:
            print(metrics.accuracy_score(y_test, y_pred))

#------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
