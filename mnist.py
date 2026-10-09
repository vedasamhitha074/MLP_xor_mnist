import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

data,labels=fetch_openml('mnist_784',version=1,return_X_y=True,as_frame=False)
data=data/255.0

train_data,test_data,train_labels,test_labels=train_test_split(data,labels,test_size=0.2,random_state=42,stratify=labels)

model=MLPClassifier(hidden_layer_sizes=(128,),activation='relu',solver='adam',max_iter=15,random_state=42)
model.fit(train_data,train_labels)

accuracy=model.score(test_data,test_labels)
print(f"Test Accuracy: {accuracy*100:.2f}%")

predictions=model.predict(test_data)
print(classification_report(test_labels,predictions))

fig,grid=plt.subplots(1,5,figsize=(10,3))
for index,image_box in enumerate(grid):
    image_box.imshow(test_data[index].reshape(28,28),cmap='gray')
    image_box.set_title(f"Pred: {predictions[index]}\nTrue: {test_labels[index]}")
    image_box.axis('off')
plt.tight_layout()
plt.show()