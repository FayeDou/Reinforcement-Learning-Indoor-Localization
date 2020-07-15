import sys
import random
import numpy as np
from keras.models import Sequential
from keras import initializers
from keras.initializers import normal, identity
from keras.initializers import VarianceScaling
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, SGD, Adam
from collections import deque
from config import Config

class DQNAgent:
    def __init__(self, state_size, action_size, config=Config()):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=config.memory_size)
        self.gamma = config.gamma   # 0.90  discount rate
        self.epsilon = config.epsilon  # exploration rate
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.learning_rate = config.learning_rate  # -4
        self.var_init = config.var_init
        self.dropout = config.dropout
        self.layers = config.layers
        self.model = self._build_model()

    def rest_epsilon_on_test(self):
        self.epsilon = 0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name), input_shape=(self.state_size,))) ##  520 + 2 + 1 + 6*20 #633 when10
        # model.add(Activation('relu'))
        model.add(Dense(self.layers[0], init=lambda shape: VarianceScaling(scale=self.var_init)(shape), input_shape=(self.state_size,)))
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout))
        # model.add(Dense(512, init=lambda shape, name: normal(shape, scale=0.01, name=name)))
        for units in self.layers[1:]:
            model.add(Dense(units, init=lambda shape: VarianceScaling(scale=self.var_init)(shape)))
            model.add(Activation('relu'))
            model.add(Dropout(self.dropout))

        model.add(Dense(self.action_size, init=lambda shape: VarianceScaling(scale=self.var_init)(shape)))
        model.add(Activation('linear'))
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model

    def choose_action(self, state):
        q_values = self.model.predict(state.T, batch_size=1)
        # print("\nq_values are : \n", q_values)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_size)
        else:
            action = np.argmax(q_values)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        train_X = []
        train_y = []
        q_eval_wrt_a=[]
        q_target=[]

        for memory in minibatch:
            old_state, action, reward, new_state, done = memory
            old_qval = self.model.predict(old_state.T, batch_size=1)
            # print(old_qval)
            new_Q = self.model.predict(new_state.T, batch_size=1)
            maxQ = np.max(new_Q)
            y = np.zeros([1, self.action_size])
            y = old_qval
            q_eval_wrt_a.append(old_qval[0, action])
            y = y.T
            # if action != self.action_size-1: #non terminal action
            update = (reward + self.gamma * maxQ)
            # else:
            #     update = reward

            q_target.append(update)
            y[action] = update

            train_X.append(old_state)
            train_y.append(y)
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        train_X = train_X.astype("float32")
        train_y = train_y.astype("float32")
        train_X = train_X[:,:,0]
        train_y = train_y[:,:,0]
        self.model.fit(train_X, train_y, batch_size=batch_size, epochs=1, verbose=2)

        # print("q_eval_wrt_a:", q_eval_wrt_a)
        # print("q_target", q_target)
        print("cost: ", np.mean((np.array(q_eval_wrt_a)-np.array(q_target))**2))
        sys.stdout.flush()

        #decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # print("Epsilon: ", self.epsilon)



