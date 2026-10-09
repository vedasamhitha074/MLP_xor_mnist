import numpy as np
np.random.seed(42)
X=np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y=np.array([
    [0],
    [1],
    [1],
    [0]
])
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
inputdim = 2
hiddendim = 2
outputdim = 1
W1=np.random.randn(inputdim,hiddendim)
b1=np.zeros((1,hiddendim))
W2=np.random.randn(hiddendim,outputdim)
b2=np.zeros((1,outputdim))
def forward_pass(X,W1,b1,W2,b2):
    z1=np.dot(X,W1)+b1
    a1=sigmoid(z1)
    z2=np.dot(a1,W2)+b2
    a2=sigmoid(z2)
    return z1,a1,z2,a2
def compute_loss(y_true,y_pred):
    return np.mean((y_true-y_pred)** 2)
lr=0.1
epochs=5000

for epoch in range(epochs):
    
    z1,a1,z2,a2=forward_pass(X,W1,b1,W2,b2)
    loss=compute_loss(y,a2)
    erroro=a2-y
    d_z2=erroro*(a2*(1.0-a2))
    dW2=np.dot(a1.T,d_z2)/X.shape[0]
    db2=np.sum(d_z2,axis=0,keepdims=True)/X.shape[0]
    errorh=np.dot(d_z2,W2.T)
    d_z1=errorh*(a1*(1.0-a1))
    dW1=np.dot(X.T,d_z1)/X.shape[0]
    db1=np.sum(d_z1,axis=0,keepdims=True)/X.shape[0]
    
    W1-=lr*dW1
    b1-=lr*db1
    W2-=lr*dW2
    b2-=lr*db2
    if (epoch+1)%2000==0:
        print(f"Epoch{epoch+1}/{epochs}-Loss:{loss:.5f}")

print("\nFinal XOR Predictions:")
z1,a1,z2,final_preds=forward_pass(X,W1,b1,W2,b2)
for i in range(len(X)):
    pred=final_preds[i][0]
    binary_class=1 if pred>0.5 else 0
    print(f"Input:{X[i]} = Target:{y[i][0]} = Raw Output:{pred:.4f} (Class:{binary_class})")

epsilon=1e-5
z1,a1,z2,a2=forward_pass(X,W1,b1,W2,b2)
erroro=a2-y
d_z2=erroro*(a2*(1.0-a2))
errorh=np.dot(d_z2, W2.T)
d_z1=errorh*(a1*(1.0-a1))
dW1_analytical=np.dot(X.T,d_z1)/X.shape[0]
analy_grad=dW1_analytical[0,0]

W1_plus=W1.copy()
W1_plus[0,0] +=epsilon
z1,a1,z2,a2_plus=forward_pass(X,W1_plus,b1,W2,b2)
loss_plus=compute_loss(y,a2_plus)
W1_minus=W1.copy()
W1_minus[0,0] -=epsilon
z1,a1,z2,a2_minus = forward_pass(X,W1_minus,b1,W2,b2)
loss_minus=compute_loss(y,a2_minus)
num_grad=(loss_plus-loss_minus)/(2*epsilon)
print(f"Analytical Gradient (Backprop) for W1[0,0]: {analy_grad:.8f}")
print(f"Numerical Gradient (Finite Diff) for W1[0,0]: {num_grad:.8f}")
grad_diff=np.abs(analy_grad-num_grad)
print(f"Absolute Difference:{grad_diff:.8e}")