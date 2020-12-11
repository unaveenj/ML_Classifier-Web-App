import streamlit as st
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

st.title("Machine Learning Web App")

st.write("""
# Choose the best ML Classifier best suited for your Dataset !
""")
dataset_chooser = st.sidebar.selectbox("Select Dataset",("Wine Dataset","Iris Dataset","Breast Cancer Dataset","Digits Dataset"))

classifier_chooser = st.sidebar.selectbox("Select classifier",("KNN","Random Forest","SVM"))

#1 get the dataset

def get_data(dataset_chooser):
    if dataset_chooser == "Iris Dataset":
        data = datasets.load_iris()
    elif dataset_chooser =="Wine Dataset":
        data=datasets.load_wine()
    elif dataset_chooser =="Breast Cancer Dataset":
        data=datasets.load_breast_cancer()
    elif dataset_chooser =="Digits Dataset":
        data=datasets.load_digits()
    X = data.data
    y = data.target
    return X,y

def parameters_ui(classifier_name):
    params = dict()
    if classifier_name == "KNN":
        K = st.sidebar.slider("K-value",1,20)
        params["K-value"]=K
    elif classifier_name=="SVM":
        C = st.sidebar.slider("C-parameter",0.01,20.0)
        params["C-parameter"] = C
    elif classifier_name=="Random Forest":
        max_depth = st.sidebar.slider("Max Depth",2,20)
        no_of_trees = st.sidebar.slider("Number of estimators ",1,100)
        params["Max Depth"]= max_depth
        params["No of estimators"] = no_of_trees
    return params

def get_classifier(classifier,params):
    if classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K-value"])
    elif classifier =="SVM":
        clf = SVC(C=params["C-parameter"])
    else:
        clf = RandomForestClassifier(n_estimators=params["No of estimators"],max_depth=params['Max Depth'],random_state=1234)
    return clf

X,y = get_data(dataset_chooser)
#Get the info of the dataset
st.write("Shape of dataset",X.shape)
st.write("Number of classes", len(np.unique(y)))
params = parameters_ui(classifier_chooser)
clf = get_classifier(classifier_chooser,params)
st.sidebar.file_uploader("hello")
#Classification
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
st.write(f"Classifier: {classifier_chooser}")
st.write(f"Accuracy: {accuracy}")

#Plot the dataset
pca = PCA(2) #2 dimensions for 2D
X_projeted = pca.fit_transform(X)
x1 = X_projeted[:,0]
x2 = X_projeted[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()





# for index,classes in enumerate(np.unique(y)):
#     st.write(f"{index}.",classes)



