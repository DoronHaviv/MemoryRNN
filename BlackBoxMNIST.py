import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import sys


dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

MinSize = 10000000

for digit in range(10):
    if(np.count_nonzero(np.argmax(dataset.train.labels,1)==digit)<MinSize):
        MinSize = np.count_nonzero(np.argmax(dataset.train.labels,1)==digit)

MNIST_Images = np.zeros((MinSize,784,10))
for digit in range(10):
    MNIST_Images[:,:,digit] = (dataset.train.images[np.where(np.argmax(dataset.train.labels,1)==digit)[0][0:MinSize]])



config = tf.ConfigProto(
      device_count={'CPU': 1, 'GPU': 4},
      allow_soft_placement=True,
      )

config.gpu_options.allow_growth = True


SysInputs=sys.argv

MNIST=True
DIGITS=False

NoiseMu = 0.1307
NoiseSig = 0.30816

def generate_batch(digit,batch_size,RandImages):
  Images=RandImages
  for i in range(batch_size):
      RandomIndexes = np.random.randint(0, MinSize - 1, 1)
      TrueImage = MNIST_Images[RandomIndexes,:,digit]
      Images[i,0] = TrueImage
  return(Images)



LearningCurr= SysInputs[2]
Arc= SysInputs[1]
timestepsSample = 15 #int(SysInputs[4])
timesteps = 15
Instance =  SysInputs[3]
BatchSize=100
num_hidden=200
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

FileName = 'RNNDynamicsPaper/MNIST/10Dig_' + LearningCurr + '_' + Arc + '_' + Instance
path = 'RNNDynamicsPaper/MNISTAttractor/BlackBoxAnalysis/' + Arc + '/' + LearningCurr + '_' + Instance + '/'

if not os.path.exists(path):
    os.makedirs(path)



HiddenStep='End'
ReadOutStep='End'
weightsFile= FileName+'/Variables/WeightsAt'+HiddenStep+'.npy'
biasesFile= FileName+'/Variables/BiasesAt'+HiddenStep+'.npy'
OutBiasesFile= FileName+'/Variables/OutBiasesAt'+HiddenStep+'.npy'
OutWeightsFile= FileName+'/Variables/OutWeightsAt'+HiddenStep+'.npy'
weights=np.load(weightsFile)
biases=np.load(biasesFile)
OutWeights=np.load(OutWeightsFile)
OutBiases=np.load(OutBiasesFile)
Weights=weights.item(0)
Biases=biases.item(0)




xRand = np.ones((784))*NoiseMu
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


X = tf.placeholder("float", [None, None ,None])
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
        xt=tf.concat((x[:,j,:],ZerosMat),axis=1)
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
        xt=tf.concat((x[:,j,:],ZerosMat),axis=1)
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

Images = np.zeros((10 * BatchSize, timesteps,784))
NoiseImages = np.random.normal(loc = NoiseMu, scale = NoiseSig/3, size = (1000,784))[0:timesteps]
FeedImages = np.repeat(NoiseImages[np.newaxis, :, :], int(BatchSize), axis=0)
for Class in range(10):
    Images[Class*BatchSize:(Class+1)*BatchSize] = generate_batch(Class, BatchSize, FeedImages)

print(Images.shape)
if(Arc=='LSTM'):
    InitHiddenH = np.zeros((10,num_hidden))
    InitHiddenC = np.zeros((10,num_hidden))
    H0, C0 = sess.run([h0,c0],feed_dict={X:Images, H_init: np.zeros((10 * BatchSize,num_hidden)), C_init: np.zeros((10 * BatchSize,num_hidden)), ZerosMat: np.zeros((10 * BatchSize,1))})

    for Digit in range(10):
        HiddenHDigit = H0[Digit*BatchSize:(Digit+1)*BatchSize]
        HiddenCDigit = C0[Digit*BatchSize:(Digit+1)*BatchSize]
        ReadOutH0 = np.argmax(ReadOutLSTM(HiddenHDigit,HiddenCDigit), axis=1) - 1
        WeightVec = (ReadOutH0 == Digit) / np.count_nonzero(ReadOutH0 == Digit)
        InitHiddenH[Digit] = np.average(HiddenHDigit, weights=WeightVec, axis=0).astype('float32')
        InitHiddenC[Digit] = np.average(HiddenCDigit, weights=WeightVec, axis=0).astype('float32')
    h = tf.Variable(InitHiddenH, dtype=tf.float32)
    c = tf.Variable(InitHiddenC, dtype=tf.float32)

if(Arc!='LSTM'):
    InitHiddenH = np.zeros((10,num_hidden))
    H0 = sess.run([h0],feed_dict={X:Images, H_init: np.zeros((10 * BatchSize,num_hidden)), ZerosMat: np.zeros((10 * BatchSize,1))})[0]
    for Digit in range(10):
        HiddenHDigit = H0[Digit*BatchSize:(Digit+1)*BatchSize]
        ReadOutH0 = np.argmax(ReadOutGRU(HiddenHDigit), axis=1) - 1
        WeightVec = (ReadOutH0 == Digit) / np.count_nonzero(ReadOutH0 == Digit)
        InitHiddenH[Digit] = np.average(HiddenHDigit, weights=WeightVec, axis=0).astype('float32')
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
#


if (Arc == 'LSTM'):
    ReadOutF20 = np.argmax(ReadOutLSTM(h_fix,c_fix), axis=1)-1
    ReadOut20 = np.argmax(ReadOutLSTM(InitHiddenH,InitHiddenC), axis=1)-1

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

np.save(path+'/Hid20.npy',InitHiddenH)
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


