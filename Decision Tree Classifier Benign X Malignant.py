import imageio
import numpy as np
from sklearn import svm
import os, os.path

#Número de imagens testadas por classe

n_imagens = 1100
passo = 1100

array_imagens = np.zeros((2*n_imagens,224*224*3), dtype=object)

#Tipo 1      

n1 = 0
n2 = n_imagens

arq_tipo1 = os.listdir('train/benign/')
tipo1 = np.zeros((224,224,3), dtype=object)

for cada_tipo1, i in zip(arq_tipo1, range(n1,n2)):
        tipo1=imageio.imread('train/benign/'+cada_tipo1) 
        matriz_tipo1 = tipo1.reshape(tipo1.shape[0]*tipo1.shape[1]*tipo1.shape[2]).T
        array_imagens[i,:] = matriz_tipo1;
        
#Tipo 2

n3 = n2
n4 = n3 + passo

arq_tipo2 = os.listdir('train/malignant/')
tipo2 = np.zeros((224,224,3), dtype=object)

for cada_tipo2, i in zip(arq_tipo2, range(n3,n4)):
        tipo2=imageio.imread('train/malignant/'+cada_tipo2)
        matriz_tipo2 = tipo2.reshape(tipo2.shape[0]*tipo2.shape[1]*tipo2.shape[2]).T
        array_imagens[i,:] = matriz_tipo2;
        
#Treino        
        
treino = np.zeros((2*n_imagens), dtype=object)

for p in range (n1,n2):    
    treino[p]='Tipo 1'
for p in range (n3,n4):    
    treino[p]='Tipo 2'
    
#Shuffle

from sklearn.utils import shuffle

array_imagens, treino = shuffle(array_imagens, treino, random_state=42)
    
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(array_imagens, treino, test_size=0.2, random_state=42)
   
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

#Prediction

predictions = dtree.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test,predictions))


# In[ ]:




