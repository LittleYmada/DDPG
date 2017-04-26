import random
from collections import deque

import keras.backend as K
import tensorflow as tf
from keras.engine.training import *
from keras.layers import Dense, Input, merge
from keras.optimizers import Adam

HIDENNEURAL1=64
HIDENNEURAL2=128
HIDENNEURAL3=256
HIDENNEURAL4=512


class QValue(object):
    def __init__(self,sess,STATE_SHAPE,ACTION_SHAPE,BATCH_SIZE,TAU,LEARNING_RATE):
        self.sess=sess
        K.set_session(sess)
        self.STATE_SHAPE=STATE_SHAPE
        self.ACTION_SHAPE=ACTION_SHAPE
        self.BATCH_SIZE=BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        self.action,self.state,self.model=self.initACNetwork()
        self.target_action,self.target_state,self.target_model=self.initACNetwork()
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradient(self,state,action):
        return self.sess.run(self.action_grads,feed_dict={
            self.action:action,
            self.state:state
        })[0]

    def initValueNetwork(self):
        act=Input(shape=[self.ACTION_SHAPE])
        sta=Input(shape=[self.STATE_SHAPE])
        act_prepro=Dense(HIDENNEURAL2,activation='relu')(act)
        sta_prepro=Dense(HIDENNEURAL2,activation='relu')(sta)
        acs_merge=merge([act_prepro,sta_prepro],mode='sum')
        fullcon1=Dense(HIDENNEURAL4,activation='relu')(acs_merge)
        fullcon2=Dense(HIDENNEURAL3,activation='relu')(fullcon1)
        outputL=Dense(1,activation='relu')(fullcon2)
        learning_model=Model(inputs=[sta,act],outputs=outputL) #model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        learning_model.compile(loss='mse', optimizer=adam)
        return act,sta,learning_model

    def updateTargetValueNetwork(self):
        value_weights=self.model.get_weights()
        target_value_weights=self.target_model.get_weights()
        for i in range(len(value_weights)):
            target_value_weights[i]=self.TAU*value_weights[i]+(1-self.TAU)*target_value_weights[i]
        self.target_model.set_weights(target_value_weights)

    '''def updateTargetAC(self):
        pass
    def ganeratorActions(self,state_array):
        pass'''
class Poliy(object):
    def __init__(self,sess,STATE_SHAPE,ACTION_SHAPE,BATCH_SIZE,TAU,LEARNING_RATE):
        self.sess=sess
        K.set_session(sess)
        self.STATE_SHAPE=STATE_SHAPE
        self.ACTION_SHAPE=ACTION_SHAPE
        self.BATCH_SIZE=BATCH_SIZE
        self.TAU=TAU
        self.LEARNING_RATE=LEARNING_RATE

        self.model,self.weights,self.state=self.initPolicyNetwork()
        self.target_model,self.target_weight,self.target_state=self.initPolicyNetwork()
        self.v_grad = tf.placeholder(tf.float32, [None, self.ACTION_SHAPE])
        self.policy_grad = tf.gradients(self.model.output,self.weights,-self.v_grad)
        grads = zip(self.policy_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def gradient(self,state,v_grad):
        return self.sess.run(self.optimize,feed_dict={
            self.v_grad:v_grad,
            self.state:state
        })

    def update_target_policy(self):
        target_policy_weights=self.target_model.get_weights()
        policy_weights=self.model.get_weights()
        for i in range(policy_weights):
            target_policy_weights[i]=self.TAU*policy_weights[i]+(1-self.TAU)*target_policy_weights[i]
        self.target_model.set_weights(target_policy_weights)

    def initPolicyNetwork(self):
        sta=Input(shape=[self.STATE_SHAPE])
        h0=Dense(HIDENNEURAL3)(sta)
        h1=Dense(HIDENNEURAL4)(h0)
        angles=Dense(self.ACTION_SHAPE,activation='tanh')(h1)
        pmodel=Model(inputs=sta,outputs=angles)
        return pmodel,pmodel.trainable_weights,sta


class ReplayBuffer(object):
    def __init__(self,BUFFER_SIZE):
        self.BUFFER_SIZE=BUFFER_SIZE
        self.replay_buffer=deque()
        self.experience_num=0

    def getbatch(self,batch_size):
        if self.experience_num<batch_size:
            return random.sample(self.replay_buffer,self.experience_num)
        else:
            return random.sample(self.replay_buffer,batch_size)

    def add(self,experience):
        while self.experience_num>=self.BUFFER_SIZE:
            self.replay_buffer.popleft()
            self.experience_num-=1
        self.replay_buffer.append(experience)
        self.experience_num+=1

    def add_sequence(self,state,action,reward,new_state,done):
        experience=[state,action,reward,new_state,done]
        self.add(experience)

    def get_count(self):
        return self.experience_num

    def erase(self):
        self.replay_buffer=deque()
        self.experience_num=0

#TODO Ornstein-Uhlenbeck Process
class OUProcess(object):
    @staticmethod
    def getNoise(x,theta,miu,sigma):
        return theta*(miu-x)+sigma*(np.random.randn(1))













