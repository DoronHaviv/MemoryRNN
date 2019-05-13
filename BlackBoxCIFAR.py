import numpy as np
import tensorflow as tf
import os
import sys
import tensorflow.contrib.keras as keras

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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
dataset.test.images, dataset.test.labels) = keras.datasets.cifar10.load_data()
dataset.train.images = dataset.train.images / 255
dataset.test.images = dataset.test.images / 255
dataset.prepare_global_attributes()
dataset.convert_labels_to_one_hot()

config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=False,
      )

config.gpu_options.allow_growth =True

SysInputs=sys.argv

MNIST=True
DIGITS=False


NoiseMu = 0.4734
NoiseSig = 0.2517

def generate_batch(digit,batch_size,T_Max,delta_min,RandImages):
  t_report_vec = np.zeros((batch_size,T_Max, 1), dtype=np.float32)
  Images=RandImages
  for i in range(batch_size):
      ArgMax=-1
      j = np.random.randint(1, dataset.test.num_examples, 1)
      if(MNIST):
          while (ArgMax!=digit):
              TrueImage = dataset.train.images[j]
              TrueLabel = dataset.train.labels[j]
              ArgMax=np.argmax(TrueLabel)
              j=j+1
      else:
          TrueImage=np.zeros((10))
          TrueImage[digit]=1
      t_read = np.random.randint(0, T_Max - delta_min, 1)
      t_report=np.random.randint(t_read+delta_min,np.minimum(t_read+T_Max,T_Max),1)
      t_report_vec[i,t_report]=np.float32(1)
      Images[i,t_read] = TrueImage
  x = Images
  return(x)




LearningCurr= SysInputs[2]
Arc= SysInputs[1]
timestepsSample = 15
timesteps = 15
Instance =  SysInputs[3]
BatchSize=1000
num_hidden=512
if(DIGITS):
    num_input=11
else:
    num_input=784+1
num_classes=11
HiddenStateDer='H'
Noise=1
LossThres=-100000
learning_rate=0.001
if(Arc=="LSTM"):
    GDSteps = 100000
else:
    GDSteps = 100000

FileName='RNNDynamicsPaper/CIFAR/10Dig_'+LearningCurr+'_'+Arc +'_' + Instance
path='RNNDynamicsPaper/CIFARAttractor/BlackBoxAnalysis/'+Arc+'/'+LearningCurr+'_' + Instance+'/'

if not os.path.exists(path):
    os.makedirs(path)



HiddenStep='End'
ReadOutStep='End'
weightsFile=FileName+'/Variables/WeightsAt'+HiddenStep+'.npy'
biasesFile=FileName+'/Variables/BiasesAt'+HiddenStep+'.npy'
OutBiasesFile=FileName+'/Variables/OutBiasesAt'+HiddenStep+'.npy'
OutWeightsFile=FileName+'/Variables/OutWeightsAt'+HiddenStep+'.npy'
weights=np.load(weightsFile)
biases=np.load(biasesFile)
OutWeights=np.load(OutWeightsFile)
OutBiases=np.load(OutBiasesFile)
Weights=weights.item(0)
Biases=biases.item(0)

W_conv1_1 = tf.constant(np.load(FileName+'/Variables/W_Conv1_1.npy'))
b_conv1_1 = tf.constant(np.load(FileName+'/Variables/b_Conv1_1.npy'))


W_conv2_1 = tf.constant(np.load(FileName+'/Variables/W_Conv2_1.npy'))
b_conv2_1 = tf.constant(np.load(FileName+'/Variables/b_Conv2_1.npy'))


W_conv3_1 = tf.constant(np.load(FileName+'/Variables/W_Conv3_1.npy'))
b_conv3_1 = tf.constant(np.load(FileName+'/Variables/b_Conv3_1.npy'))


W_fc1 = tf.constant(np.load(FileName+'/Variables/W_fc1.npy'))
b_fc1 = tf.constant(np.load(FileName+'/Variables/b_fc1.npy'))

W_fc2 = tf.constant(np.load(FileName+'/Variables/W_fc2.npy'))
b_fc2 = tf.constant(np.load(FileName+'/Variables/b_fc2.npy'))

W_fc3 = tf.constant(np.load(FileName+'/Variables/W_fc3.npy'))
b_fc3 = tf.constant(np.load(FileName+'/Variables/b_fc3.npy'))


xColor = np.random.normal(loc = NoiseMu, scale = NoiseSig, size = (1000,32,32,3)).astype('float32')
def CNN(ColorImage):
    x_image = tf.reshape(ColorImage, [-1, 32, 32, 3])
    h_conv1 = tf.nn.dropout(tf.nn.leaky_relu(conv2d(x_image, W_conv1_1) + b_conv1_1), keep_prob=1.00)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.dropout(tf.nn.leaky_relu(conv2d(h_pool1, W_conv2_1) + b_conv2_1), keep_prob=1.00)
    h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.dropout(tf.nn.leaky_relu(conv2d(h_pool2, W_conv3_1) + b_conv3_1), keep_prob=1.00)
    h_pool3 = max_pool_2x2(h_conv3)

    #
    #
    h_pool3_flat = tf.reshape(h_pool3, [-1, 2048])
    h_fc1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1), keep_prob=1.00)
    h_fc2 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(h_fc1, W_fc2) + b_fc2), keep_prob=1.00)
    h_fc3 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(h_fc2, W_fc3) + b_fc3), keep_prob=1.00)
    return (h_fc3)


xRandTensor =CNN(xColor)

with tf.Session(config = config) as sess1:
    xRand = np.mean(sess1.run([xRandTensor])[0],axis=0)
xt_Rand = np.concatenate((xRand, [0]), axis=0)
xt_Rand = np.reshape(xt_Rand, [1, -1]).astype(np.float32)
xReadOut = np.concatenate((xRand, [1]), axis=0)
xReadOut = np.reshape(xReadOut, [1, -1]).astype(np.float32)

if(Arc=='LSTM'):
    weights = {'Wf': tf.constant(Weights['Wf']),
               'Uf': tf.constant(Weights['Uf']),
               'Wi': tf.constant(Weights['Wi']),
               'Ui': tf.constant(Weights['Ui']),
               'Wo': tf.constant(Weights['Wo']),
               'Uo': tf.constant(Weights['Uo']),
               'Wc': tf.constant(Weights['Wc']),
               'Uc': tf.constant(Weights['Uc'])}

    biases = {'bf': tf.constant(Biases['bf']),
              'bi': tf.constant(Biases['bi']),
              'bo': tf.constant(Biases['bo']),
              'bc': tf.constant(Biases['bc'])}

if (Arc == 'MGRU'):
    weights = {'Wf': tf.constant(Weights['Wf']),
               'Uf': tf.constant(Weights['Uf']),
               'Wh': tf.constant(Weights['Wh']),
               'Uh': tf.constant(Weights['Uh'])}

    biases = {'bf': tf.constant(Biases['bf']),
              'bh': tf.constant(Biases['bh'])}

if (Arc == 'GRU'):
    weights = {'Wz': tf.constant(Weights['Wz']),
               'Uz': tf.constant(Weights['Uz']),
               'Wr': tf.constant(Weights['Wr']),
               'Ur': tf.constant(Weights['Ur']),
               'Wh': tf.constant(Weights['Wh']),
               'Uh': tf.constant(Weights['Uh'])}

    biases = {'bz': tf.constant(Biases['bz']),
              'br': tf.constant(Biases['br']),
              'bh': tf.constant(Biases['bh'])}


out_weights = tf.constant(OutWeights)
out_biases = tf.constant(OutBiases)


X = tf.placeholder("float", [None, None ,None, None ,None])
ZerosMat =  tf.placeholder("float", [None ,None])
H_init = tf.placeholder("float", [None,None])

if(Arc == 'LSTM'):
    C_init = tf.placeholder("float", [None,None])

def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def ReadOutLSTM(STATEH, STATEC):
    STATEF = sigmoid(np.matmul(xReadOut, Weights['Wf'])+np.matmul(STATEH, Weights['Uf'])+Biases['bf'])
    STATEI = sigmoid(np.matmul(xReadOut, Weights['Wi']) + np.matmul(STATEH, Weights['Ui']) + Biases['bi'])
    STATEO = sigmoid(np.matmul(xReadOut, Weights['Wo'])+np.matmul(STATEH, Weights['Uo'])+Biases['bo'])
    STATEC = STATEF*STATEC + STATEI*np.tanh(np.matmul(xReadOut, Weights['Wc'])+np.matmul(STATEH, Weights['Uc'])+Biases['bc'])
    STATEH = STATEO * np.tanh(STATEC)
    return(np.matmul(STATEH, OutWeights)+OutBiases)

def ReadOutMGRU(STATEH):
    STATEF = sigmoid(np.matmul(xReadOut, Weights['Wf'])+np.matmul(STATEH, Weights['Uf'])+Biases['bf'])
    STATEH = STATEF*STATEH+(1-STATEF)*np.tanh(np.matmul(xReadOut, Weights['Wh'])+np.matmul(STATEF*STATEH, Weights['Uh'])+Biases['bh'])
    return(np.matmul(STATEH, OutWeights)+OutBiases)


def ReadOutGRU(STATEH):
    STATER = sigmoid(np.matmul(xReadOut, Weights['Wr'])+np.matmul(STATEH, Weights['Ur'])+Biases['br'])
    STATEZ = sigmoid(np.matmul(xReadOut, Weights['Wz']) + np.matmul(STATEH, Weights['Uz']) + Biases['bz'])
    STATEH = (1-STATEZ)*STATEH+STATEZ*np.tanh(np.matmul(xReadOut, Weights['Wh'])+np.matmul(STATER*STATEH, Weights['Uh'])+Biases['bh'])
    return(np.matmul(STATEH, OutWeights)+OutBiases)

def SigmaGTag(Vec):
    return(tf.sigmoid(Vec)*(1-tf.sigmoid(Vec)))

def SigmaHTag(Vec):
    return(1-tf.square(tf.tanh(Vec)))



def LSTMRNN(x,TS,Ts):
    i=0
    for j in range(Ts, TS):
        xt=tf.concat((CNN(x[:,j,:,:,:]),ZerosMat),axis=1)
        WfXt = tf.matmul(xt, weights['Wf'])
        WiXt = tf.matmul(xt, weights['Wi'])
        WoXt = tf.matmul(xt, weights['Wo'])
        WcXt = tf.matmul(xt, weights['Wc'])
        if(i==0):
            Ufht = tf.matmul(H_init, weights['Uf'])
            Uiht = tf.matmul(H_init,weights['Ui'])
            Uoht = tf.matmul(H_init,weights['Uo'])
            Ucht = tf.matmul(H_init,weights['Uc'])
            ct = C_init
        else:
            Ufht = tf.matmul(ht,weights['Uf'])
            Uiht = tf.matmul(ht,weights['Ui'])
            Uoht = tf.matmul(ht,weights['Uo'])
            Ucht = tf.matmul(ht,weights['Uc'])
        ft=tf.sigmoid(WfXt+Ufht+biases['bf'])
        it=tf.sigmoid(WiXt+Uiht+biases['bi'])
        ot=tf.sigmoid(WoXt+Uoht+biases['bo'])
        ct=tf.multiply(ft,ct)+tf.multiply(it,tf.tanh(WcXt+Ucht+biases['bc']))
        ht=tf.multiply(ot,tf.tanh(ct))
        i=i+1
    return(ht, ct)


def GRURNN(x,TS, Ts):
    i=0
    for j in range(Ts, TS):
        xt=tf.concat((CNN(x[:,j,:,:,:]),ZerosMat),axis=1)
        WzXt = tf.matmul(xt, weights['Wz'])
        WrXt = tf.matmul(xt, weights['Wr'])
        WhXt = tf.matmul(xt, weights['Wh'])
        if(i==0):
            Uzht = tf.matmul(H_init, weights['Uz'])
            Urht = tf.matmul(H_init, weights['Ur'])
            zt = tf.sigmoid(WzXt + Uzht + biases['bz'])
            rt = tf.sigmoid(WrXt + Urht + biases['br'])
            Uhrtht = tf.matmul(tf.multiply(rt, H_init), weights['Uh'])
            ht = tf.add(tf.multiply(1 - zt, H_init), tf.multiply(zt, tf.tanh(WhXt + Uhrtht + biases['bh'])))
        else:
            Uzht = tf.matmul(ht, weights['Uz'])
            Urht = tf.matmul(ht, weights['Ur'])
            zt = tf.sigmoid(WzXt + Uzht + biases['bz'])
            rt = tf.sigmoid(WrXt + Urht + biases['br'])
            Uhrtht = tf.matmul(tf.multiply(rt, ht), weights['Uh'])
            ht = tf.add(tf.multiply(1 - zt, ht), tf.multiply(zt, tf.tanh(WhXt + Uhrtht + biases['bh'])))
        i=i+1
    return (ht)

if(Arc=='LSTM'):
    h0, c0 = LSTMRNN(X,timestepsSample,0)
if (Arc == 'GRU'):
    h0 = GRURNN(X,timestepsSample,0)

init1 = tf.global_variables_initializer()

sess=tf.Session(config = config)
sess.run(init1)

NoiseImages = np.random.normal(loc = NoiseMu, scale = NoiseSig, size = (timesteps,32,32,3)).astype('float32')
FeedImages = np.repeat(NoiseImages[np.newaxis, :, :, :, :], int(BatchSize), axis=0)

if(Arc=='LSTM'):
    InitHiddenH = np.zeros((10,num_hidden))
    InitHiddenC = np.zeros((10,num_hidden))

    for Digit in range(10):
        Images = generate_batch(Digit, BatchSize, timesteps, timesteps - 1, FeedImages)
        H0, C0 = sess.run([h0, c0], feed_dict={X: Images, H_init: np.zeros((BatchSize, num_hidden)), C_init: np.zeros((BatchSize, num_hidden)), ZerosMat: np.zeros((BatchSize, 1))})
        HiddenHDigit = H0
        HiddenCDigit = C0
        ReadOutH0 = np.argmax(ReadOutLSTM(HiddenHDigit,HiddenCDigit), axis=1) - 1
        WeightVec = (ReadOutH0 == Digit) / np.count_nonzero(ReadOutH0 == Digit)
        InitHiddenH[Digit] = np.average(HiddenHDigit, weights=WeightVec, axis=0).astype('float32')
        InitHiddenC[Digit] = np.average(HiddenCDigit, weights=WeightVec, axis=0).astype('float32')
    h = tf.Variable(InitHiddenH, dtype=tf.float32)
    c = tf.Variable(InitHiddenC, dtype=tf.float32)

if(Arc!='LSTM'):
    InitHiddenH = np.zeros((10,num_hidden))

    for Digit in range(10):
        Images = generate_batch(Digit, BatchSize, timesteps, timesteps - 1, FeedImages)
        H0 = sess.run([h0], feed_dict={X: Images, H_init: np.zeros((BatchSize, num_hidden)),ZerosMat: np.zeros((BatchSize, 1))})[0]
        HiddenHDigit = H0
        ReadOutH0 = np.argmax(ReadOutGRU(HiddenHDigit), axis=1) - 1
        WeightVec = (ReadOutH0 == Digit) / np.count_nonzero(ReadOutH0 == Digit)
        InitHiddenH[Digit] = np.average(HiddenHDigit, axis=0, weights=WeightVec).astype('float32')
    h = tf.Variable(InitHiddenH, dtype=tf.float32)

xt_rand = tf.constant(xt_Rand)
LearnRate = tf.placeholder(tf.float32, [])
if(Arc=='LSTM'):
    f = tf.sigmoid(tf.matmul(xt_rand, weights['Wf']) + tf.matmul(h, weights['Uf']) + biases['bf'])
    i = tf.sigmoid(tf.matmul(xt_rand, weights['Wi']) + tf.matmul(h, weights['Ui']) + biases['bi'])
    o = tf.sigmoid(tf.matmul(xt_rand, weights['Wo']) + tf.matmul(h, weights['Uo']) + biases['bo'])
    cNew = tf.multiply(f, c) + tf.multiply(i, tf.tanh(tf.matmul(xt_rand, weights['Wc']) + tf.matmul(h, weights['Uc']) + biases['bc']))
    hDer = tf.multiply(o, tf.tanh(cNew))-h
    cDer = tf.tanh(cNew) - tf.tanh(c)
    loss = tf.reduce_sum(tf.square(hDer) + tf.square(cDer))
    SpeedEach = tf.reduce_sum(tf.square(hDer) + tf.square(cDer),axis=1)
    optimizer = tf.train.AdamOptimizer(learning_rate=LearnRate)
    train_op = optimizer.minimize(loss)
    clip_op = tf.assign(h, tf.clip_by_value(h, -1, 1))


if(Arc=='GRU'):
    z = tf.sigmoid(tf.matmul(xt_rand, weights['Wz']) + tf.matmul(h, weights['Uz']) + biases['bz'])
    r = tf.sigmoid(tf.matmul(xt_rand, weights['Wr']) + tf.matmul(h, weights['Ur']) + biases['br'])
    hDer = tf.add(tf.multiply(1-z, h), tf.multiply(z, tf.tanh(tf.matmul(xt_rand, weights['Wh']) + tf.matmul(tf.multiply(r, h), weights['Uh']) + biases['bh'])))-h
    loss= tf.reduce_sum(tf.square(hDer))
    SpeedEach = tf.reduce_sum(tf.square(hDer),axis=1)
    optimizer = tf.train.AdamOptimizer(learning_rate=LearnRate)
    train_op = optimizer.minimize(loss,var_list=[h])
    clip_op = tf.assign(h, tf.clip_by_value(h, -1, 1))


init2 = tf.global_variables_initializer()

sess.run(init2)
LossP100=1000
for j in range(GDSteps):
    if(j==40000):
        learning_rate = learning_rate/10
    if(Arc == 'LSTM'):
        _, lossP, h_fix, c_fix,SpeedDig = sess.run([train_op, loss, h, c, SpeedEach], feed_dict={LearnRate: learning_rate})
    else:
        _, lossP, h_fix, SpeedDig=sess.run([train_op,loss,h,SpeedEach],feed_dict={LearnRate: learning_rate})
    sess.run(clip_op)
    if(j%100==0):
        LossP100=lossP
        print(np.sqrt(lossP))


if (Arc == 'LSTM'):

    ReadOutF20 = np.argmax(ReadOutLSTM(h_fix,c_fix), axis=1)-1
    ReadOut20 = np.argmax(ReadOutLSTM(InitHiddenH,InitHiddenC), axis=1)-1

    print('\nReadOutFF_20: ' + str(ReadOutF20))
    print('ReadOut_20: ' + str(ReadOut20))

if (Arc == 'MGRU'):
    ReadOutF20 = np.argmax(ReadOutMGRU(h_fix), axis=1)-1
    ReadOut20 = np.argmax(ReadOutMGRU(InitHiddenH), axis=1)-1

    print('\nReadOutFF_20: ' + str(ReadOutF20))
    print('ReadOut_20: ' + str(ReadOut20))

if (Arc == 'GRU'):

    ReadOutF20 = np.argmax(ReadOutGRU(h_fix), axis=1)-1
    ReadOut20 = np.argmax(ReadOutGRU(InitHiddenH), axis=1)-1

    print('\nReadOutFF_20: ' + str(ReadOutF20))
    print('ReadOut_20: ' + str(ReadOut20))


print('h20-h20F: ' + str(np.sqrt(np.sum(np.square(InitHiddenH-h_fix)))))


if(Arc == 'LSTM'):
    np.save(path + '/CHid20.npy', InitHiddenC)
    np.save(path + '/CHidF20.npy', c_fix)
    np.save(path + '/FinalCHid.npy', c_fix)

np.save(path+'/Hid20.npy',H0)
np.save(path+'/HidF20.npy',h_fix)
np.save(path+'/FinalHid.npy',h_fix)

file = open(path + '/EigVal.txt', 'w')


np.save(path + '/FinalSpeed', np.sqrt(SpeedDig))
file.write('\n\nReadOutFF_20: ' + str(ReadOutF20))
np.save(path + '/ReadOutF20', ReadOutF20)
file.write('\nReadOut_20: ' + str(ReadOut20))
np.save(path + '/ReadOut_20', ReadOut20)

file.write('h20-h20F: ' + str(np.sqrt(np.sum(np.square(InitHiddenH - h_fix)))))
file.write('\nHdif: ' + str(np.sqrt(np.sum(np.square(InitHiddenH - h_fix)))))
file.write("\nFinal Hidden: " + str(h_fix))
file.close()


