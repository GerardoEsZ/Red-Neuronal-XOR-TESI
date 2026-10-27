"""Motor MLP 2-2-1 para XOR. Implementación explícita de backpropagation con NumPy."""
import numpy as np
X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = np.array([[0.],[1.],[1.],[0.]])

def sigmoid(z):
    return 1.0/(1.0+np.exp(-np.clip(z,-500,500)))

class RedXOR:
    def __init__(self, seed=17):
        rng=np.random.default_rng(seed)
        self.W1=rng.normal(0,1,(2,2)); self.b1=np.zeros((1,2))
        self.W2=rng.normal(0,1,(2,1)); self.b2=np.zeros((1,1))
        self.epoch=0
        self.historial=[]
    def forward(self, Xdata=X):
        x=np.asarray(Xdata,dtype=float).reshape(-1,2)
        z1=x@self.W1+self.b1; a1=sigmoid(z1)
        z2=a1@self.W2+self.b2; a2=sigmoid(z2)
        return {'x':x,'z1':z1,'a1':a1,'z2':z2,'a2':a2}
    def loss(self):
        p=self.forward()['a2']
        return float(-np.mean(Y*np.log(np.clip(p,1e-12,1))+(1-Y)*np.log(np.clip(1-p,1e-12,1))))
    def step(self, lr=1.0):
        f=self.forward(); n=len(X)
        # Entropía cruzada binaria + sigmoide: delta2 = (y_pred-y)/N
        delta2=(f['a2']-Y)/n
        gW2=f['a1'].T@delta2; gb2=delta2.sum(axis=0,keepdims=True)
        delta1=(delta2@self.W2.T)*f['a1']*(1-f['a1'])
        gW1=X.T@delta1; gb1=delta1.sum(axis=0,keepdims=True)
        before=self.loss()
        old={'W1':self.W1.copy(),'b1':self.b1.copy(),'W2':self.W2.copy(),'b2':self.b2.copy()}
        self.W2-=lr*gW2; self.b2-=lr*gb2
        self.W1-=lr*gW1; self.b1-=lr*gb1
        self.epoch+=1; after=self.loss(); self.historial.append(after)
        return {'antes':before,'despues':after,'grad_W1':gW1,'grad_b1':gb1,'grad_W2':gW2,'grad_b2':gb2,'old':old}
    def train(self,lr=1.0,epochs=10000):
        for _ in range(epochs): self.step(lr)
        return self
    def metrics(self):
        p=self.forward()['a2'].reshape(-1)
        return {'loss':self.loss(),'accuracy':float(np.mean((p>=.5)==Y.reshape(-1))),'prob':p}

if __name__=='__main__':
    for rate in [.1,1.0]:
        nn=RedXOR().train(rate,10000)
        print('Tasa',rate,'Loss',nn.loss(),'Exactitud',nn.metrics()['accuracy'],'Probabilidades',nn.metrics()['prob'])
