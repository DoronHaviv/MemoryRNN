import numpy as np
import tensorflow as tf
import sys
import time
import os




map_fn = tf.map_fn

SysInputs=sys.argv

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

batch_size = 64
display_step = 10
TINY=0.0001


# Network Parameters
num_input = 1025 # MNIST data input (img shape: 28*28)
timesteps = 20 #int(SysInputs[1]) # timesteps
t_max = 20 #int(SysInputs[1])
delta_t_min= 4 #int(SysInputs[3]) # minimal difference between t_r/home/alexander/doronNet/TriggerMNIST.py:90ead and t_report
num_hidden = 512 # hidden layer num of features
num_classes = 11  # MNIST total classes (0-9 digits)

MaxClass =  11
LearningCurr = SysInputs[2]
Architechture = SysInputs[1]
Instance =  SysInputs[3]
Mode = SysInputs[4]

if(Mode == 'Init'):
    SaveFileName = 'RNNDynamicsPaper/CIFAR_SpeedReg/Init/' + '10Dig_' + str(LearningCurr) + '_' + Architechture + '_' + Instance
    RegPointsH = 'HidF20'
    RegPointsC = 'CHidF20'
    Lambda = 0.0
if(Mode == 'Control'):
    SaveFileName = 'RNNDynamicsPaper/CIFAR_SpeedReg/Control/' + '10Dig_' + str(LearningCurr) + '_' + Architechture + '_' + Instance
    RegPointsH = 'HidF20'
    RegPointsC = 'CHidF20'
    Lambda = 0.0
if(Mode == 'SlowPoints'):
    SaveFileName = 'RNNDynamicsPaper/CIFAR_SpeedReg/SlowPoints/' + '10Dig_' + str(LearningCurr) + '_' + Architechture + '_' + Instance
    RegPointsH = 'HidF20'
    RegPointsC = 'CHidF20'
    Lambda = 0.01
if(Mode == 'CenterMass'):
    SaveFileName = 'RNNDynamicsPaper/CIFAR_SpeedReg/CenterMass/' + '10Dig_' + str(LearningCurr) + '_' + Architechture + '_' + Instance
    RegPointsH = 'Hid20'
    RegPointsC = 'CHid20'
    Lambda = 0.01




if(Mode!='Init'):
    InitFile = 'RNNDynamicsPaper/CIFAR_SpeedReg/Init/' + '10Dig_' + str(LearningCurr) + '_' + Architechture + '_' + Instance
    InitHiddenStep = 'End'
    InitReadOutStep = 'End'
    InitweightsFile =  InitFile + '/Variables/WeightsAt' + InitHiddenStep + '.npy'
    InitbiasesFile = InitFile + '/Variables/BiasesAt' + InitHiddenStep + '.npy'
    InitOutBiasesFile = InitFile + '/Variables/OutBiasesAt' + InitReadOutStep + '.npy'
    InitOutWeightsFile =  InitFile + '/Variables/OutWeightsAt' + InitReadOutStep + '.npy'
    Initweights = np.load(InitweightsFile)
    Initbiases = np.load(InitbiasesFile)
    InitOutWeights = np.load(InitOutWeightsFile)
    InitOutBiases = np.load(InitOutBiasesFile)
    Initweights = Initweights.item(0)
    Initbiases = Initbiases.item(0)

    NumRep = 10

    NumRep = 10
    slowpointsH = np.zeros((10*NumRep,512))


    for i in range(10):
        slowpointsH[int(i*NumRep):int((i+1)*NumRep)] = np.repeat(np.reshape(np.load('RNNDynamicsPaper/CIFARAttractor/BlackBoxAnalysis/' + Architechture + '/' + LearningCurr + '_' + str(Instance) + '/'+RegPointsH+'.npy')[i],[1, -1]),NumRep,axis =0)

    SlowPointH = tf.constant(slowpointsH.astype(np.float32))

    if(Architechture == 'LSTM'):
        slowpointsC = np.zeros((10*NumRep, 512))
        for i in range(10):
            slowpointsC[int(i*NumRep):int((i+1)*NumRep)] =  np.repeat(np.reshape(np.load('RNNDynamicsPaper/CIFARAttractor/BlackBoxAnalysis/'  + Architechture + '/' + LearningCurr + '_' + str(Instance) + '/'+RegPointsC+'.npy')[i],[1, -1]),NumRep,axis =0)
        SlowPointC = tf.constant(slowpointsC.astype(np.float32))



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

if(Mode!='Init'):

    learning_rate = 0.0001
    training_steps = 5000

    ZeroWeight = np.ones((training_steps, 1))
    MaxClasses = np.ones((training_steps, 1)) * MaxClass
    T_MAX_VEC = np.ones((training_steps, 1)) * t_max

    W_conv1_1 = tf.Variable(np.load(InitFile+'/Variables/W_Conv1_1.npy'))
    b_conv1_1 = tf.Variable(np.load(InitFile+'/Variables/b_Conv1_1.npy'))

    W_conv2_1 = tf.Variable(np.load(InitFile+'/Variables/W_Conv2_1.npy'))
    b_conv2_1 = tf.Variable(np.load(InitFile+'/Variables/b_Conv2_1.npy'))

    W_conv3_1 = tf.Variable(np.load(InitFile+'/Variables/W_Conv3_1.npy'))
    b_conv3_1 = tf.Variable(np.load(InitFile+'/Variables/b_Conv3_1.npy'))

    W_fc1 = tf.Variable(np.load(InitFile+'/Variables/W_fc1.npy'))
    b_fc1 = tf.Variable(np.load(InitFile+'/Variables/b_fc1.npy'))

    W_fc2 = tf.Variable(np.load(InitFile+'/Variables/W_fc2.npy'))
    b_fc2 = tf.Variable(np.load(InitFile+'/Variables/b_fc2.npy'))

    W_fc3 = tf.Variable(np.load(InitFile+'/Variables/W_fc3.npy'))
    b_fc3 = tf.Variable(np.load(InitFile+'/Variables/b_fc3.npy'))

    if(USE_LSTM):

        weights = {'Wf' : tf.Variable(Initweights['Wf']),
        'Uf' : tf.Variable(Initweights['Uf']),
        'Wi' : tf.Variable(Initweights['Wi']),
        'Ui' : tf.Variable(Initweights['Ui']),
        'Wo' : tf.Variable(Initweights['Wo']),
        'Uo' : tf.Variable(Initweights['Uo']),
        'Wc' : tf.Variable(Initweights['Wc']),
        'Uc' : tf.Variable(Initweights['Uc'])}

        biases ={'bf' : tf.Variable(Initbiases['bf']),
        'bi' : tf.Variable(Initbiases['bi']),
        'bo' : tf.Variable(Initbiases['bo']),
        'bc' : tf.Variable(Initbiases['bc'])}

    if(USE_GRU):

        weights = { 'Wh' : tf.Variable(Initweights['Wh']),
                    'Uh' : tf.Variable(Initweights['Uh']),
                    'Wz' : tf.Variable(Initweights['Wz']),
                    'Uz' : tf.Variable(Initweights['Uz']),
                    'Wr' : tf.Variable(Initweights['Wr']),
                    'Ur' : tf.Variable(Initweights['Ur'])}

        biases = {'bh' : tf.Variable(Initbiases['bh']),
                  'bz' : tf.Variable(Initbiases['bz']),
                  'br': tf.Variable(Initbiases['br'])}



    out_weights = tf.Variable(InitOutWeights)
    out_biases = tf.Variable(InitOutBiases)

else:

    learning_rate = 0.0001
    training_steps = 300000

    ZeroWeight = np.ones((training_steps, 1))
    MaxClasses = np.ones((training_steps, 1)) * MaxClass
    T_MAX_VEC = np.ones((training_steps, 1)) * t_max

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


ZerosVec = tf.zeros([NumRep*10,1])

def LSTMSpeed(HiddenH,HiddenC):
    RandImages = dist.sample([NumRep*10, 32, 32, 3])
    xt_rand = tf.concat((CNN(RandImages),ZerosVec), axis = 1)
    f = tf.sigmoid(tf.matmul(xt_rand, weights['Wf']) + tf.matmul(HiddenH, weights['Uf']) + biases['bf'])
    i = tf.sigmoid(tf.matmul(xt_rand, weights['Wi']) + tf.matmul(HiddenH, weights['Ui']) + biases['bi'])
    o = tf.sigmoid(tf.matmul(xt_rand, weights['Wo']) + tf.matmul(HiddenH, weights['Uo']) + biases['bo'])
    cNew = tf.multiply(f, HiddenC) + tf.multiply(i, tf.tanh(tf.matmul(xt_rand, weights['Wc']) + tf.matmul(HiddenH, weights['Uc']) + biases['bc']))
    hDer = tf.multiply(o, tf.tanh(cNew))-HiddenH
    cDer = tf.tanh(cNew) - tf.tanh(HiddenC)
    Speed = tf.reduce_sum(tf.square(hDer) + tf.square(cDer))
    return(Speed)

def GRUSpeed(HiddenH):
    RandImages = dist.sample([NumRep*10, 32, 32, 3])
    xt_rand = tf.concat((CNN(RandImages),ZerosVec), axis = 1)
    z = tf.sigmoid(tf.matmul(xt_rand, weights['Wz']) + tf.matmul(HiddenH, weights['Uz']) + biases['bz'])
    r = tf.sigmoid(tf.matmul(xt_rand, weights['Wr']) + tf.matmul(HiddenH, weights['Ur']) + biases['br'])
    hDer = tf.add(tf.multiply(1-z, HiddenH), tf.multiply(z, tf.tanh(tf.matmul(xt_rand, weights['Wh']) + tf.matmul(tf.multiply(r, HiddenH), weights['Uh']) + biases['bh'])))-HiddenH
    Speed = tf.reduce_sum(tf.square(hDer))
    return(Speed)


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

if(USE_LSTM):
    SpeedLoss = LSTMSpeed(SlowPointH,SlowPointC)  #Loss on Digits and zeros
    #loss=tf.reduce_mean((t_max-1)*Z*Y*tf.log(prediction+TINY)+alpha*Y*tf.log(prediction+TINY)) - Lambda * L
else:
    SpeedLoss = GRUSpeed(SlowPointH)
    #loss=tf.reduce_mean((t_max-1)*Z*Y*tf.log(prediction+TINY)+alpha*Y*tf.log(prediction+TINY)) - Lambda *
SoftMaxLoss = - tf.reduce_mean((t_max-1)*Z*Y*tf.log(prediction+TINY)+alpha*Y*tf.log(prediction+TINY))
loss= SoftMaxLoss + Lambda * SpeedLoss
optimizer = tf.train.AdamOptimizer(learning_rate=LearnRate)
train_op1 = optimizer.minimize(loss)

# Evaluate model (with test logits, for dropout to be disabled)

# Initialize the variables (i.e. assign their default value)
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
        _,pred,Rate,Weights,Biases,OutWeights,OutBiases,Speed, SFLoss  = sess.run([train_op1,prediction,loss,weights,biases,out_weights,out_biases, SpeedLoss, SoftMaxLoss], feed_dict={X: batch_x, Y: batch_y,Z: batch_z,W:batch_w ,alpha: ZeroWeight[step-1]
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
            print("Step " + str(step) + ", Total Success " + str(TotalAcc) + ", Digits Success " + str(DigitAcc), ' Speed: ', Speed, ' SoftMaxLoss: ', SFLoss, ' time:', str(t_next - t_this))
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
