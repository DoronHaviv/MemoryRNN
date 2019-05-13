import numpy as np
import tensorflow as tf
import sys
import time
import os




map_fn = tf.map_fn

SysInputs=sys.argv

FASHION='C'
Noise=1

class Image_dataset:
    class train:
        pass

    class test:
        pass

    def prepare_global_attributes(self):
        self.train.num_examples = len(self.train.images)
        self.test.num_examples = len(self.test.images)
        self.image_size = self.test.images[0].size
        self.image_shape = self.test.images[0].shape

    def convert_labels_to_one_hot(self):
        label_span = np.array(list(range(1 + np.max(self.train.labels))))
        self.train.labels = dataset.train.labels == label_span
        self.test.labels = dataset.test.labels == label_span


dataset = Image_dataset()
(dataset.train.images, dataset.train.labels), (
dataset.test.images, dataset.test.labels) = tf.keras.datasets.cifar10.load_data()
dataset.train.images = dataset.train.images / 255
dataset.test.images = dataset.test.images  / 255
dataset.prepare_global_attributes()
dataset.convert_labels_to_one_hot()
num_input = 1 + dataset.image_size
NoiseMu = 0.4734
NoiseSig = 0.2517


path=""



MinSize = 100000

for digit in range(10):
    if(np.count_nonzero(np.argmax(dataset.train.labels,1)==digit)<MinSize):
        MinSize = np.count_nonzero(np.argmax(dataset.train.labels,1)==digit)

CIFARImages = np.zeros((MinSize,32,32,3,10))
for digit in range(10):
    CIFARImages[:,:,:,:,digit] = (dataset.train.images[np.where(np.argmax(dataset.train.labels,1)==digit)[0][0:MinSize]])

def generate_batch(batch_size,T_Max,TimeSteps,delta_min,NumClasses,MaxClasses):
  Images=np.zeros((batch_size, TimeSteps, 32,32,3))
  t_read_mat = np.ones((batch_size, TimeSteps, 32, 32 ,3))
  t_report_vec = np.zeros((batch_size,TimeSteps, 1), dtype=np.float32)
  t_error_mat = np.zeros((batch_size,NumClasses,TimeSteps), dtype=np.float32)
  OnesMat= -1000*np.ones((batch_size,NumClasses,TimeSteps), dtype=np.float32)
  OnesMat[:,0:MaxClasses,:]=1
  Labels=np.zeros((batch_size,NumClasses,TimeSteps))
  Labels[:,0,0:T_Max] =1
  ReportTime = np.zeros((batch_size,T_Max))
  i=0
  for i in range(batch_size):
      RandomIndex = np.random.randint(0, MinSize - 1, 1)
      RandomDigit = np.random.randint(0, MaxClasses-1, 1)
      TrueImage = CIFARImages[RandomIndex, :, :, :, RandomDigit]
      TrueLabel = np.zeros(NumClasses)
      TrueLabel[RandomDigit+1]=1
      t_read=np.random.randint(0,T_Max-delta_min,1)
      t_report=np.random.randint(t_read+delta_min,T_Max,1)
      t_report_vec[i,t_report]=np.float32(1)
      t_error_mat[i, 0:MaxClasses ,t_report] = np.ones((1, MaxClasses), dtype=np.float32)
      Labels[i,:,t_report]=TrueLabel
      Labels[i, :, T_Max:TimeSteps] = 0
      Images[i, t_read] = TrueImage
      ReportTime[i,t_report] = 1
      t_read_mat[i, t_read, :] = 0
  x=Images
  y=Labels
  z=t_error_mat
  return(x,t_report_vec,y,z,OnesMat,t_read_mat,ReportTime)



#File Name Importatant!!!!!




# Training Parameters
learning_rate = 0.0001
training_steps = 300000
batch_size = 64
display_step = 10
graph_step=100
save_step=1000000
TINY=0.0001
OnlyOutLim=2


# Network Parameters
num_input = 1025 # MNIST data input (img shape: 28*28)
timesteps = 20 #int(SysInputs[1]) # timesteps
t_max = 20 #int(SysInputs[1])
delta_t_min= 4 #int(SysInputs[3]) # minimal difference between t_r/home/alexander/doronNet/TriggerMNIST.py:90ead and t_report
num_hidden = 512 # hidden layer num of features
num_classes = 11  # MNIST total classes (0-9 digits)

MaxClass =  11
Instance = SysInputs[3]
LearningCurr = SysInputs[2]
Architechture=  SysInputs[1]


MirrorPrecent=0
Partial='Full'

USE_CURR=True

SaveFileName='RNNDynamicsPaper/CIFAR/'+'10Dig_'+str(LearningCurr)+'_'+Architechture + '_' + Instance

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

    if (LearningCurr == 'DeCu'):
        T_MAX_VEC[0:100000] = 6
        T_MAX_VEC[100000:120000] = 8
        T_MAX_VEC[120000:140000] = 10
        T_MAX_VEC[140000:160000] = 12
        T_MAX_VEC[160000:180000] = 14
        T_MAX_VEC[180000:200000] = 16
        T_MAX_VEC[200000:220000] = 18
        T_MAX_VEC[220000:training_steps] = t_max

    if (LearningCurr == 'VoCu'):
        MaxClasses[0:30000] = 3
        MaxClasses[30000:60000] = 4
        MaxClasses[60000:90000] = 5
        MaxClasses[90000:120000] = 6
        MaxClasses[120000:150000] = 7
        MaxClasses[150000:180000] = 8
        MaxClasses[180000:210000] = 9
        MaxClasses[210000:240000] = 10
        MaxClasses[240000:training_steps] = num_classes


#with tf.device('/device:GPU:0'):

# tf Graph input
X = tf.placeholder("float", [None,None ,32 , 32 ,3])
TRep = tf.placeholder("float", [None,None ,1])
Y = tf.placeholder("float", [None,num_classes ,None])
Z = tf.placeholder("float", [None,num_classes ,None])
W = tf.placeholder("float", [None,num_classes ,None])
T_ReadVec = tf.placeholder("float", [None,None ,32, 32 , 3])
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


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=1.0)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

num_filter1 = 128
num_filter2 = 128
num_filter3 = 128

W_conv1_1 = weight_variable([3, 3, 3, num_filter1])/np.sqrt(3 * num_filter1)
b_conv1_1 = bias_variable([num_filter1]) / np.sqrt(num_filter1)

W_conv2_1 = weight_variable([3, 3, num_filter1, num_filter2])/np.sqrt(num_filter1 * num_filter2)
b_conv2_1 = bias_variable([num_filter2])/np.sqrt(num_filter2)

W_conv3_1 = weight_variable([3, 3, num_filter2, num_filter3])/np.sqrt(num_filter2 * num_filter3)
b_conv3_1 = bias_variable([num_filter3])/np.sqrt(num_filter3)

W_fc1 = tf.Variable(np.random.randn(2048, 2048).astype(np.float32) / np.sqrt(2048))
b_fc1 = tf.Variable(0.01 * np.random.randn(2048).astype(np.float32))

W_fc2 = tf.Variable(np.random.randn(2048, 2048).astype(np.float32) / np.sqrt(2048))
b_fc2 = tf.Variable(0.01 * np.random.randn(2048).astype(np.float32))

W_fc3 = tf.Variable(np.random.randn(2048, 1024).astype(np.float32) / np.sqrt(2048))
b_fc3 = tf.Variable(0.01 * np.random.randn(1024).astype(np.float32))

def CNN(ColorImage):
    x_image = tf.reshape(ColorImage, [-1, 32, 32, 3])
    h_conv1 = tf.nn.dropout(tf.nn.leaky_relu(conv2d(x_image, W_conv1_1) + b_conv1_1), keep_prob = 1.00)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.dropout(tf.nn.leaky_relu(conv2d(h_pool1, W_conv2_1) + b_conv2_1), keep_prob = 1.00)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.dropout(tf.nn.leaky_relu(conv2d(h_pool2, W_conv3_1) + b_conv3_1), keep_prob = 1.00)
    h_pool3 = max_pool_2x2(h_conv3)

    h_pool3_flat = tf.reshape(h_pool3, [-1, 2048])
    h_fc1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1), keep_prob = 1.00)
    h_fc2 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(h_fc1, W_fc2) + b_fc2), keep_prob = 1.00)
    h_fc3 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(h_fc2, W_fc3) + b_fc3), keep_prob = 1.00)
    return(h_fc3)



def LSTMRNN(x):
    x = tf.reshape(x, shape = [batch_size*timesteps,32,32,3])
    T_ReadVecReshape = tf.reshape(T_ReadVec, shape = [batch_size*timesteps,32,32,3])
    Noise = dist.sample([batch_size*timesteps, 32, 32, 3])
    x = CNN(x[:,:,:,:]+ T_ReadVecReshape[:,:,:,:]*Noise)
    x = tf.reshape(x, shape = [batch_size,timesteps,1024])
    for j in range(timesteps):
        xt=tf.concat((x[:,j,:,],TRep[:,j,:]),axis=1)
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
        OutPut_TimeStep=tf.nn.sigmoid(tf.matmul(tf.nn.dropout(ht,keep_prob=1.0), out_weights) + out_biases)
        OutPut_TimeStep=tf.reshape(OutPut_TimeStep, (batch_size, num_classes, 1))
        if (j == 0):
            output =OutPut_TimeStep
        else:
            output = tf.concat((output,OutPut_TimeStep), 2)
    return(output)

def GRURNN(x):
    x = tf.reshape(x, shape = [batch_size*timesteps,32,32,3])
    T_ReadVecReshape = tf.reshape(T_ReadVec, shape = [batch_size*timesteps,32,32,3])
    Noise = dist.sample([batch_size*timesteps, 32, 32, 3])
    x = CNN(x[:,:,:,:]+ T_ReadVecReshape[:,:,:,:]*Noise)
    x = tf.reshape(x, shape = [batch_size,timesteps,1024])
    for j in range(timesteps):
        xt=tf.concat((x[:,j,:,],TRep[:,j,:]),axis=1)
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
        OutPut_TimeStep= tf.nn.sigmoid(tf.matmul(tf.nn.dropout(ht,keep_prob=1.0), out_weights) + out_biases)
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
train_op1 = optimizer.minimize(-loss)
init = tf.global_variables_initializer()


t_this=time.time()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
# Start training
with tf.Session(config=config) as sess:
    # Run the initializer
    sess.run(init)
    TotalAccList = []
    DigitAccList = []
    print("STARTING OPTIMIZATION")
    TrainOutOn = 0 #Train on hidden and readout
    for step in range(1, training_steps+1):

        if(step==280000):
            learning_rate = learning_rate/10


        SumHits = 0
        SumNonezero = 0
        SumHitsIndvDig=np.zeros((num_classes-1,1))
        SumIndvDig=np.zeros((num_classes-1,1))
        SumHitZero=0
        SumTotalZero=0
        batch_x, TReport, batch_y, batch_z, batch_w,InputTimes, RepTimes = generate_batch(batch_size,int(T_MAX_VEC[step-1]),timesteps,delta_t_min,num_classes,int(MaxClasses[step-1]))
        _,pred,Rate,Weights,Biases,OutWeights,OutBiases = sess.run([train_op1,prediction,loss,weights,biases,out_weights,out_biases], feed_dict={X: batch_x, Y: batch_y,Z: batch_z,W:batch_w ,alpha: ZeroWeight[step-1]
                                                                                                                                                                ,LearnRate: learning_rate, T_ReadVec: InputTimes, TRep: TReport})

        if step % display_step == 0 or step == 1:
            pred = pred[:,:,0:int(T_MAX_VEC[step-1])]
            batch_y = batch_y[:,:,0:int(T_MAX_VEC[step-1])]
            predictions = np.argmax(pred,axis=1)
            labels = np.argmax(batch_y, axis=1)

            predictionsTrigger = predictions[RepTimes.astype('int32') == 1]
            labelsTrigger = labels[RepTimes.astype('int32') == 1]

            DigitAcc = np.sum(predictionsTrigger == labelsTrigger) / batch_size
            TotalAcc = np.sum(predictions == labels) / (batch_size*int(T_MAX_VEC[step-1]))
            TotalAccList.append(TotalAcc)
            DigitAccList.append(DigitAcc)
            t_next = time.time()
            print("Step " + str(step) + ", Total Success " + str(TotalAcc) + ", Digits Success " + str(DigitAcc), ' time:', str(t_next - t_this))
            t_this = t_next

    print("Optimization Finished!")

    np.save(path+SaveFileName+"/Success/TotalHitRate", TotalAccList)
    np.save(path+SaveFileName+"/Success/DigitsHitRate", DigitAccList)

    np.save(path+SaveFileName+"/Variables/WeightsAtEnd", Weights)
    np.save(path+SaveFileName+"/Variables/OutWeightsAtEnd", OutWeights)
    np.save(path+SaveFileName+"/Variables/BiasesAtEnd", Biases)
    np.save(path+SaveFileName+"/Variables/OutBiasesAtEnd", OutBiases)

    np.save(path+SaveFileName+"/Variables/W_Conv1_1", W_conv1_1.eval())
    np.save(path+SaveFileName+"/Variables/b_Conv1_1", b_conv1_1.eval())

    np.save(path+SaveFileName+"/Variables/W_Conv2_1", W_conv2_1.eval())
    np.save(path+SaveFileName+"/Variables/b_Conv2_1", b_conv2_1.eval())
    #
    np.save(path+SaveFileName+"/Variables/W_Conv3_1", W_conv3_1.eval())
    np.save(path+SaveFileName+"/Variables/b_Conv3_1", b_conv3_1.eval())

    np.save(path+SaveFileName+"/Variables/W_fc1", W_fc1.eval())
    np.save(path+SaveFileName+"/Variables/b_fc1", b_fc1.eval())

    np.save(path+SaveFileName+"/Variables/W_fc2", W_fc2.eval())
    np.save(path+SaveFileName+"/Variables/b_fc2", b_fc2.eval())

    np.save(path+SaveFileName+"/Variables/W_fc3", W_fc3.eval())
    np.save(path+SaveFileName+"/Variables/b_fc3", b_fc3.eval())
