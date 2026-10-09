import numpy as np

X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])
np.random.seed(42)

w1=np.random.randn(2,2)
b1=np.zeros((1,2))
w2=np.random.randn(2,1)
b2=np.zeros((1,1))

lr=0.1
epochs=5000

for i in range(epochs):
    z1=np.dot(X,w1)+b1
    a1=1/(1+np.exp(-z1))
    z2=np.dot(a1,w2)+b2
    a2=1/(1+np.exp(-z2))

    loss=np.mean((a2-y)**2)

    dz2=(a2-y)*(a2*(1-a2))
    dw2=np.dot(a1.T,dz2)/4
    db2=np.sum(dz2,axis=0,keepdims=True)/4

    dz1=np.dot(dz2,w2.T)*(a1*(1-a1))
    dw1=np.dot(X.T,dz1)/4
    db1=np.sum(dz1,axis=0,keepdims=True)/4

    w1-=lr*dw1
    b1-=lr*db1
    w2-=lr*dw2
    b2-=lr*db2

    if (i+1)%1000==0:
        print("epoch",i+1,"loss:",round(loss,5))

print("\nFinal XOR Predictions:")
print(np.round(a2,3))

eps=1e-4

w1_p=w1.copy()
w1_p[0,0]+=eps
a2_p=1/(1+np.exp(-(np.dot(1/(1+np.exp(-(np.dot(X,w1_p)+b1))),w2)+b2)))
l_p=np.mean((a2_p-y)**2)

w1_m=w1.copy()
w1_m[0,0]-=eps
a2_m=1/(1+np.exp(-(np.dot(1/(1+np.exp(-(np.dot(X,w1_m)+b1))),w2)+b2)))
l_m=np.mean((a2_m-y)**2)

num_g=(l_p-l_m)/(2*eps)
print("\nGradient check diff for w1[0,0]:",abs(dw1[0,0]-num_g))


