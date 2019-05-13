
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import tensorflow.contrib.layers as layers
import sys
import os
import time
import scipy.stats as stats
from tensorflow.contrib import rnn



map_fn = tf.map_fn

SysInputs=sys.argv


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
dataset = mnist
num_input = 784+1
num_hidden=200
NoiseMu = 0.1307
NoiseSig = 0.30816



path=""

config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      )

config.gpu_options.allow_growth =True

NoiseMu = 0.1307
NoiseSig = 0.30816

TruncMu = 0.37
TruncSig = 0.9237

alphaTr, betaTr = (0 - TruncMu) / TruncSig, (1 - TruncMu) / TruncSig



MinSize = 100000

for digit in range(10):
    if(np.count_nonzero(np.argmax(dataset.train.labels,1)==digit)<MinSize):
        MinSize = np.count_nonzero(np.argmax(dataset.train.labels,1)==digit)

MNIST_Images = np.zeros((MinSize,784,10))
for digit in range(10):
    MNIST_Images[:,:,digit] = (dataset.train.images[np.where(np.argmax(dataset.train.labels,1)==digit)[0][0:MinSize]])

def generate_batch(batch_size,T_Max,TimeSteps,delta_min,NumClasses,MaxClasses):
  Images=np.zeros((batch_size, TimeSteps, 784))
  t_read_mat = np.concatenate((np.ones((batch_size, TimeSteps, 784)), np.zeros((batch_size, TimeSteps, 1))), axis=2)
  t_report_vec = np.zeros((batch_size,TimeSteps, 1), dtype=np.float32)
  t_error_mat = np.zeros((batch_size,NumClasses,TimeSteps), dtype=np.float32)
  OnesMat= -1000*np.ones((batch_size,NumClasses,TimeSteps), dtype=np.float32)
  OnesMat[:,0:MaxClasses,:]=1
  Labels=np.zeros((batch_size,NumClasses,TimeSteps))
  Labels[:,0,0:T_Max] =1
  i=0
  for i in range(batch_size):
      RandomIndex = np.random.randint(0, MinSize - 1, 1)
      RandomDigit = np.random.randint(0, MaxClasses-1, 1)
      TrueImage = MNIST_Images[RandomIndex, :, RandomDigit]
      TrueLabel = np.zeros(NumClasses)
      TrueLabel[RandomDigit+1]=1
      t_read=np.random.randint(0,T_Max-delta_min,1)
      t_report=np.random.randint(t_read+delta_min,T_Max,1)
      t_report_vec[i,t_report]=np.float32(1)
      t_error_mat[i, 0:MaxClasses ,t_report] = np.ones((1, MaxClasses), dtype=np.float32)
      Labels[i,:,t_report]=TrueLabel
      Labels[i, :, T_Max:TimeSteps] = 0
      Images[i, t_read] = TrueImage
      t_read_mat[i, t_read, :] = 0
  x=np.concatenate((Images,t_report_vec),axis=2)
  y=Labels
  z=t_error_mat
  return(x,y,z,OnesMat,t_read_mat)



#File Name Importatant!!!!!




# Training Parameters
learning_rate = 0.00001
training_steps = 140000
batch_size = 128
display_step = 10
TINY=0.0001


# Network Parameters
num_input = 785 # MNIST data input (img shape: 28*28)
timesteps = 20 #int(SysInputs[1]) # timesteps
t_max = 20 #int(SysInputs[1])
delta_t_min= 4#int(SysInputs[3]) # minimal difference between t_r/home/alexander/doronNet/TriggerMNIST.py:90ead and t_report
num_hidden = 200 # hidden layer num of features
num_classes = 11  # MNIST total classes (0-9 digits)

MaxClass =  11
MirrorPrecent=0
Partial='Full'
Instance = SysInputs[3]
LearningCurr = SysInputs[2]
Architechture=  SysInputs[1]


    #Timesteps = 20


USE_CURR=True
SaveFileName='RNNDynamicsPaper/MNIST/'+'10Dig_'+str(LearningCurr)+'_'+Architechture +'_' + Instance


NewFolder = path+SaveFileName
NewFolderVariables=path+SaveFileName+"/Variables"
NewFolderSuccess=path+SaveFileName+"/Success"
NewFolderGraphs=path+SaveFileName+"/Graphs"

if not os.path.exists(NewFolder):
    os.makedirs(NewFolder)
    os.makedirs(NewFolderVariables)
    os.makedirs(NewFolderSuccess)
    os.makedirs(NewFolderGraphs)

if(Architechture=='LSTM'):
    USE_GRU=False
    USE_LSTM=True
elif(Architechture=='GRU'):
    USE_GRU=True
    USE_LSTM=False


ZeroWeight=np.ones((training_steps,1))
MaxClasses=np.ones((training_steps,1))*MaxClass
T_MAX_VEC=np.ones((training_steps,1))*t_max

if(USE_CURR):
    if(LearningCurr=='ReCu'):
        ZeroWeight[0:20000]=0
        ZeroWeight[20000:30000]=0.25
        ZeroWeight[30000:40000]=0.5
        ZeroWeight[40000:50000]=0.75
        ZeroWeight[50000:60000] = 0.75
        ZeroWeight[60000:training_steps]=1

    if(LearningCurr=='DeCu'):
        T_MAX_VEC[0:20000] = 6
        T_MAX_VEC[20000:30000] = 8
        T_MAX_VEC[30000:40000] = 10
        T_MAX_VEC[40000:50000] = 12
        T_MAX_VEC[50000:60000] = 14
        T_MAX_VEC[60000:70000] = 16
        T_MAX_VEC[70000:80000] = 18
        T_MAX_VEC[80000:training_steps] = t_max

    if(LearningCurr=='VoCu'):
        MaxClasses[0:8000] = 3
        MaxClasses[8000:16000] = 4
        MaxClasses[16000:26000] = 5
        MaxClasses[26000:36000] = 6
        MaxClasses[36000:48000] = 7
        MaxClasses[48000:62000] = 8
        MaxClasses[62000:78000] = 9
        MaxClasses[78000:98000] = 10
        MaxClasses[98000:training_steps]= num_classes

# tf Graph input
X = tf.placeholder("float", [None,None ,num_input])
Y = tf.placeholder("float", [None,num_classes ,None])
Z = tf.placeholder("float", [None,num_classes ,None])
W = tf.placeholder("float", [None,num_classes ,None])
T_ReadVec = tf.placeholder("float", [None,None,None])
alpha=tf.placeholder("float", [1])
LearnRate = tf.placeholder(tf.float32, [])
# Define weights


if(USE_LSTM):

    weights = {'Wf': tf.Variable(np.random.randn(num_input, num_hidden).astype(np.float32) / np.sqrt(num_input)),
               'Uf': tf.Variable(np.random.randn(num_hidden, num_hidden).astype(np.float32) / np.sqrt(num_hidden)),
               'Wi': tf.Variable(np.random.randn(num_input, num_hidden).astype(np.float32) / np.sqrt(num_input)),
               'Ui': tf.Variable(np.random.randn(num_hidden, num_hidden).astype(np.float32) / np.sqrt(num_hidden)),
               'Wo': tf.Variable(np.random.randn(num_input, num_hidden).astype(np.float32) / np.sqrt(num_input)),
               'Uo': tf.Variable(np.random.randn(num_hidden, num_hidden).astype(np.float32) / np.sqrt(num_hidden)),
               'Wc': tf.Variable(np.random.randn(num_input, num_hidden).astype(np.float32) / np.sqrt(num_input)),
               'Uc': tf.Variable(np.random.randn(num_hidden, num_hidden).astype(np.float32) / np.sqrt(num_hidden))}

    biases = {'bf': tf.Variable(0.01 * np.random.randn(num_hidden).astype(np.float32)),
              'bi': tf.Variable(0.01 * np.random.randn(num_hidden).astype(np.float32)),
              'bo': tf.Variable(0.01 * np.random.randn(num_hidden).astype(np.float32)),
              'bc': tf.Variable(0.01 * np.random.randn(num_hidden).astype(np.float32))}

if(USE_GRU):

    weights = {'Wh': tf.Variable(np.random.randn(num_input, num_hidden).astype(np.float32) / np.sqrt(num_input)),
               'Uh': tf.Variable(np.random.randn(num_hidden, num_hidden).astype(np.float32) / np.sqrt(num_hidden)),
               'Wz': tf.Variable(np.random.randn(num_input, num_hidden).astype(np.float32) / np.sqrt(num_input)),
               'Uz': tf.Variable(np.random.randn(num_hidden, num_hidden).astype(np.float32) / np.sqrt(num_hidden)),
               'Wr': tf.Variable(np.random.randn(num_input, num_hidden).astype(np.float32) / np.sqrt(num_input)),
               'Ur': tf.Variable(np.random.randn(num_hidden, num_hidden).astype(np.float32) / np.sqrt(num_hidden))}

    biases = {'bh': tf.Variable(0.01 * np.random.randn(num_hidden).astype(np.float32)),
              'bz': tf.Variable(0.01 * np.random.randn(num_hidden).astype(np.float32)),
              'br': tf.Variable(0.01 * np.random.randn(num_hidden).astype(np.float32))}



out_weights = tf.Variable(np.random.randn(num_hidden,num_classes).astype(np.float32)/np.sqrt(num_hidden))
out_biases = tf.Variable(0.01*np.random.randn(num_classes).astype(np.float32))

dist = tf.contrib.distributions.Normal(loc = NoiseMu, scale = NoiseSig)
#dist = tf.contrib.distributions.Uniform(low = 0.0, high = 1.0)



def LSTMRNN(x):
    for j in range(timesteps):
        Noise = dist.sample([batch_size,num_input])
        xt=x[:,j,:] + T_ReadVec[:,j,:]*Noise
        WfXt = tf.matmul(xt, weights['Wf'])
        WiXt = tf.matmul(xt, weights['Wi'])
        WoXt = tf.matmul(xt, weights['Wo'])
        WcXt = tf.matmul(xt, weights['Wc'])
        if(j==0):
            ft=tf.sigmoid(WfXt+biases['bf'])
            it=tf.sigmoid(WiXt+biases['bi'])
            ot=tf.sigmoid(WoXt+biases['bo'])
            ct=tf.multiply(it,tf.tanh(WcXt+biases['bc']))
            ht=tf.multiply(ot,tf.tanh(ct))
        else:
            Ufht=tf.matmul(ht,weights['Uf'])
            Uiht=tf.matmul(ht,weights['Ui'])
            Uoht=tf.matmul(ht,weights['Uo'])
            Ucht=tf.matmul(ht,weights['Uc'])
            ft=tf.sigmoid(WfXt+Ufht+biases['bf'])
            it=tf.sigmoid(WiXt+Uiht+biases['bi'])
            ot=tf.sigmoid(WoXt+Uoht+biases['bo'])
            ct=tf.multiply(ft,ct)+tf.multiply(it,tf.tanh(WcXt+Ucht+biases['bc']))
            ht=tf.multiply(ot,tf.tanh(ct))
        OutPut_TimeStep = tf.sigmoid(tf.matmul(ht, out_weights) + out_biases)
        OutPut_TimeStep = tf.reshape(OutPut_TimeStep, (batch_size, num_classes, 1))
        if (j == 0):
            output = OutPut_TimeStep
        else:
            output = tf.concat((output, OutPut_TimeStep), 2)
    return (output)

def GRURNN(x):
    for j in range(timesteps):
        Noise = dist.sample([batch_size,num_input])
        xt=x[:,j,:] + T_ReadVec[:,j,:]*Noise
        WrXt = tf.matmul(xt, weights['Wr'])
        WzXt = tf.matmul(xt, weights['Wz'])
        WhXt = tf.matmul(xt, weights['Wh'])
        if(j==0):
            zt = tf.sigmoid(WzXt + biases['bz'])
            rt = tf.sigmoid(WrXt + biases['br'])
            ht=tf.multiply(zt,tf.tanh(WhXt+biases['bh']))
        else:
            Uzht = tf.matmul(ht, weights['Uz'])
            Urht = tf.matmul(ht, weights['Ur'])
            zt = tf.sigmoid(WzXt + Uzht + biases['bz'])
            rt = tf.sigmoid(WrXt + Urht + biases['br'])
            Uhrtht=tf.matmul(tf.multiply(rt,ht),weights['Uh'])
            ht = tf.add(tf.multiply(1-zt,ht),tf.multiply(zt, tf.tanh(WhXt +Uhrtht+ biases['bh'])))
        OutPut_TimeStep=tf.sigmoid(tf.matmul(ht, out_weights) + out_biases)
        OutPut_TimeStep=tf.reshape(OutPut_TimeStep, (batch_size, num_classes, 1))
        if (j == 0):
            output =OutPut_TimeStep
        else:
            output = tf.concat((output,OutPut_TimeStep), 2)
    return(output)


if(USE_LSTM):
    logits = LSTMRNN(X)
if(USE_GRU):
    logits = GRURNN(X)
logitsLoss=logits*W
prediction = tf.nn.softmax(logitsLoss,1)
logits_arg_max=tf.argmax(logits,1)
labels_arg_max=tf.argmax(Y,1)
max_in_logits=tf.reduce_max(logits,1)

# Define loss and optimizer

loss=tf.reduce_mean((t_max-1)*Z*Y*tf.log(prediction+TINY)+alpha*Y*tf.log(prediction+TINY)) #Loss on Digits and zeros

optimizer = tf.train.AdamOptimizer(learning_rate=LearnRate)
train_op1 = optimizer.minimize(-loss)  # train hidden and readout
# Evaluate model (with test logits, for dropout to be disabled)






# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add saver


t_this=time.time()

# Start training
with tf.Session(config=config) as sess:
    # Run the initializer
    sess.run(init)
    hitNotZerolist=[]
    hitZeroList=[]
    TotalHitRateList=[]
    print("STARTING OPTIMIZATION")
    for step in range(1, training_steps+1):

        if(step==120000):
            learning_rate = learning_rate/10


        SumHits = 0
        SumNonezero = 0
        SumHitsIndvDig=np.zeros((num_classes-1,1))
        SumIndvDig=np.zeros((num_classes-1,1))
        SumHitZero=0
        SumTotalZero=0
        batch_x, batch_y, batch_z, batch_w,InputTimes = generate_batch(batch_size,int(T_MAX_VEC[step-1]),timesteps,delta_t_min,num_classes,int(MaxClasses[step-1]))
        _,pred,Rate,Weights,Biases,OutWeights,OutBiases = sess.run([train_op1,prediction,loss,weights,biases,out_weights,out_biases], feed_dict={X: batch_x, Y: batch_y,Z: batch_z,W:batch_w ,alpha: ZeroWeight[step-1]
                                                                                                                                                                ,LearnRate: learning_rate, T_ReadVec: InputTimes})

        pred=np.swapaxes(pred,1,2)
        batch_y = np.swapaxes(batch_y, 1, 2)
        batch_z=np.swapaxes(batch_z, 1, 2)

        pred=np.reshape(pred,(batch_size*timesteps,num_classes))
        batch_y=np.reshape(batch_y,(batch_size*timesteps,num_classes))
        batch_z=np.reshape(batch_z,(batch_size*timesteps,num_classes))


        pred=np.argmax(pred,1)
        Labels_arg_max = np.argmax(batch_y, 1)
        batch_z_max = np.max(batch_z, 1)


        for index in range(pred.shape[0]):
            if(index%timesteps<int(T_MAX_VEC[step-1])):
                if(batch_z_max[index] != 0):
                    if(pred[index] == Labels_arg_max[index]):
                        SumHits = SumHits+1
                        SumHitsIndvDig[Labels_arg_max[index]-1][0] = SumHitsIndvDig[Labels_arg_max[index]-1][0] + 1
                    SumNonezero = SumNonezero+1
                    SumIndvDig[Labels_arg_max[index]-1][0] = SumIndvDig[Labels_arg_max[index]-1][0] + 1
                else:
                    if(pred[index] == Labels_arg_max[index]):
                        SumHitZero=SumHitZero+1
                    SumTotalZero=SumTotalZero+1


        Hit_NotZero_Rate = SumHits / SumNonezero
        Hit_Rate_IndvDig=SumHitsIndvDig/SumIndvDig
        ZeroHitRate= SumHitZero/SumTotalZero
        TotalHitRate=round((SumHitZero+SumHits)/(SumTotalZero+SumNonezero),2)


        hitNotZerolist.append(Hit_NotZero_Rate)
        hitZeroList.append(ZeroHitRate)
        TotalHitRateList.append(TotalHitRate)
        if(step==1):
            hit_list_IndvDig=Hit_Rate_IndvDig
        else:
            hit_list_IndvDig = np.concatenate((hit_list_IndvDig,Hit_Rate_IndvDig),axis=1)


        if step % display_step == 0 or step==1:
            t_next = time.time()
            print("Step " + str(step)+", Total Success "+str(TotalHitRate)+", Digits Success " + str(Hit_NotZero_Rate),' time:', str(t_next-t_this))
            t_this = t_next


    print("Optimization Finished!")

    np.save(path+SaveFileName+"/Variables/WeightsAtEnd", Weights)
    np.save(path+SaveFileName+"/Variables/OutWeightsAtEnd", OutWeights)
    np.save(path+SaveFileName+"/Variables/BiasesAtEnd", Biases)
    np.save(path+SaveFileName+"/Variables/OutBiasesAtEnd", OutBiases)

    np.save(path+SaveFileName+"/Success/TotalHitRate", TotalHitRateList)
    np.save(path+SaveFileName+"/Success/ZeroHitRate", hitZeroList)
    np.save(path+SaveFileName+"/Success/DigitsHitRate", hitNotZerolist)
