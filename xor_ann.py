#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


xor_train_data = np.array([[0, 0],
                           [0, 1],
                           [1, 0],
                           [1, 1]])

xor_train_output = np.array([[0],
                             [1],
                             [1],
                             [0]])


# In[3]:


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):

        self.hidden_input_weights = np.random.rand(input_size, hidden_size) # Generates values from 0 to 1 randomly
        self.hidden_output_weights = np.random.rand(hidden_size, output_size)

    def sigmoid(self, x):

        return 1/(1+np.exp(-x))

    def derivate_sigmoid(self, x):

        return x*(1-x)

    def train(self, train_data, train_output, epochs):

        for i in range(epochs):

            #Forward Propagation

            hidden_output = self.sigmoid(np.dot(train_data, self.hidden_input_weights))
            output_output = self.sigmoid(np.dot(hidden_output, self.hidden_output_weights))

            #Backward Propagation

            output_error = train_output-output_output
            self.hidden_output_weights += np.dot(hidden_output.T, output_error*self.derivate_sigmoid(output_output))
            
            hidden_error = np.dot(output_error*self.derivate_sigmoid(output_output), self.hidden_output_weights.T)
            self.hidden_input_weights += np.dot(train_data.T, hidden_error*self.derivate_sigmoid(hidden_output))

    def predict(self, test_data):

        hidden_output = self.sigmoid(np.dot(test_data, self.hidden_input_weights))
        return self.sigmoid(np.dot(hidden_output, self.hidden_output_weights))   


# In[4]:


model = NeuralNetwork(2, 5, 1)


# In[7]:


model.train(xor_train_data, xor_train_output, 10000)


# In[8]:


threshold = 0.5


# In[9]:


predictions = model.predict(xor_train_data)
final_predictions = []


# In[10]:


for i in predictions:

    if i > threshold: final_predictions.append(1)
    else: final_predictions.append(0)


# In[11]:


print("Input\tOutput")

for i in range(4):

    print(xor_train_data[i],"--->",final_predictions[i])


# In[ ]:




