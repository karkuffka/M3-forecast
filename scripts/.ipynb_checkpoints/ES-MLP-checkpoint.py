
# coding: utf-8

# In[1]:


import pandas as pd
import dynet as dy
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time


# In[3]:


df = pd.read_excel('../data/international-airline-passengers.xls',header=None).iloc[15:]
df.columns = ['Time','P1']
df.set_index('Time',drop=True,inplace=True)
df = df*1

print(df.shape)


# In[42]:


plt.plot(df)


# In[67]:


df = pd.Series(np.arange(1,200))/10+40
plt.plot(df)
noise = np.random.normal(0,5,(199))
df+=noise
plt.plot(df)
#s = 1+0.3*np.sin(2*np.pi*np.arange(1,120)/12)
#plt.plot(df*s)
#df = (df*s+100).to_frame()
df = df.to_frame()


# In[54]:


class Forecaster(object):
    # The init method adds parameters to the parameter collection.
    def __init__(self, pc,input_len):
        self.W = []
        self.W.append(pc.add_parameters((2, input_len), init='normal', mean=0.5,std = 0.1))
        self.bias = pc.add_parameters((2), init='normal',mean=0,std = 0.1)
        self.W.append(pc.add_parameters((1, 2), init='normal', mean=0.5,std = 0.1))
    # the __call__ method applies the network to an input
    def __call__(self, x):
        y = self.W[0]*x+self.bias
        y = self.W[1]*y
        #y = dy.rectify(y)
        return y
m = dy.ParameterCollection()


# In[68]:


SERIES = df.shape[1]

dy.renew_cg()
m = dy.ParameterCollection()
alphas = m.add_parameters((SERIES,1), init=0.5)#alpha smoothing
gammas = m.add_parameters((SERIES,1), init=0.)#seasonal smoothing
seasonals = m.add_parameters((12,SERIES),init='normal', mean=1.2, std=0.2)#seasonal component

n = 2 #number of inputs
h = 1 #pred horizontal

#forecasting net
fcstr = Forecaster(m,n)
y = dy.inputTensor(df.values)



levelSm = dy.logistic(alphas) #ensure positive

#seasonal model!!!!--------------------------
seasonSm = dy.logistic(gammas) #ensure positive
seasonInit =  dy.rectify(seasonals)+0.001 #make sure no zeros are there


#no seasonal model!!!!-----------------------
#block seasonality
# seasonInit = dy.ones((12,1))
# seasonSm = dy.zeros(1,1)




#fill first 12 seasonal values
newSeason = [seasonInit[i] for i in range(12)]
newLevels = []

#Like Smyl's
newSeason.append(newSeason[0])
newLevels.append(1*dy.cdiv(y[0],newSeason[0]))
#perform smoothing
for i in range(1,len(df)):
    newLevels.append(levelSm*dy.cdiv(y[i],newSeason[i])+(1-levelSm)*newLevels[i-1])
    newSeason.append(seasonSm*dy.cdiv(y[i],newLevels[i])+(1-seasonSm)*newSeason[i])
s = dy.concatenate(newSeason)
l = dy.concatenate(newLevels)



#penalize sudden level changes (should be scale independent - it is dependent)\
#should penalize 2nd derivative
l_log_diff = dy.log(dy.cdiv(l[1:],l[0:l.dim()[0][0]-1]))
l_penalty = l_log_diff[1:]-l_log_diff[0:l_log_diff.dim()[0][0]-1]
level_loss = dy.mean_elems(dy.square(l_penalty))*10
print(level_loss.value())

preds = []
outputs=[]

#wez y i usun sezonowosc i level
for i in range(n,len(df)-h):
    inputs = y[i-n:i]#n okresy
    curr_season = s[i-n:i]
    inputs = dy.cdiv(inputs,l[i])
    inputs = dy.cdiv(inputs,curr_season)
    inputs = dy.log(inputs)
    reseasonalize = s[i+1]#poprzedni okres +1 krok
    preds.append(dy.exp(fcstr(inputs))*l[i]*reseasonalize)          
    outputs.append(y[i+1])#+1 krok
predictions = dy.concatenate(preds)
outputs = dy.concatenate(outputs)

#log_err = dy.mean_elems(dy.abs(dy.log(outputs)-dy.log(predictions)))
err =dy.mean_elems(dy.abs(outputs-predictions))
loss = err + level_loss

trainer = dy.SimpleSGDTrainer(m,learning_rate = 0.25)

loss_value = loss.value()
print(seasonInit.npvalue())
for i in range(2000):
    loss.backward()
    trainer.update()
    loss_value = loss.value(recalculate=True)
    trainer.learning_rate *=0.992 #0.99 lr decay
    
    if i%50==0 or i<10:
        
        print("the mae after step is:",err.value(recalculate=True))
        print("the loss after step is:",loss_value)
        print('l_loss1: ',level_loss.value(recalculate=True))
        plt.plot(predictions.value(recalculate=True),label = 'Pred')
        plt.plot(l.value(recalculate=True),label = 'lvl')
        plt.plot(outputs.value(),label ='gt')
        plt.show()

print('levelSm ',levelSm.npvalue())
print('seaosonSm ',seasonSm.npvalue())
print('init season ',seasonInit.npvalue())

plt.plot(l.value(recalculate=True))
plt.title('Level')
plt.show()
plt.plot(s.value(recalculate=True))
plt.title('Seasonal')
plt.show()


# In[64]:


from pandas.plotting import autocorrelation_plot
#autocorrelation_plot(pd.Series(l.value(recalculate=True)[:,0]))
# plt.show()
autocorrelation_plot(df)
autocorrelation_plot(l.value())

