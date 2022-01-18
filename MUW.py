#80-20

import numpy as np
import scipy as scipy
from qiskit import *
from qiskit.tools.jupyter import *
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
import sys
import time
import traceback
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects #parallelization with 'multiprocessing'

from sklearn.metrics.pairwise import rbf_kernel

from qiskit.tools.visualization import plot_bloch_multivector

# to run on real quantum computer
#from qiskit import IBMQ
#job_monitor('job_id')
#from qiskit.tools.monitor import job_monitor
#IBMQ.enable_account('')
#provider = IBMQ.get_provider(hub='ibm-q')

#backend = provider.get_backend('ibmq_london') # Specifying Quantum device

backend = Aer.get_backend('qasm_simulator')


#############################################################################################################


# Preprocessing #

#rotation angles
def get_angles(x):
#beta_{js}
    beta11 = 2 * np.arcsin(np.sqrt(x[1]) ** 2 / np.sqrt(x[0] ** 2 + x[1] ** 2))
    beta21 = 2 * np.arcsin(np.sqrt(x[3]) ** 2 / np.sqrt(x[2] ** 2 + x[3] ** 2))
    beta31 = 2 * np.arcsin(np.sqrt(x[5]) ** 2 / np.sqrt(x[4] ** 2 + x[5] ** 2))
    beta41 = 2 * np.arcsin(np.sqrt(x[7]) ** 2 / np.sqrt(x[6] ** 2 + x[7] ** 2))
    beta51 = 2 * np.arcsin(np.sqrt(x[9]) ** 2 / np.sqrt(x[8] ** 2 + x[9] ** 2))
    beta61 = 2 * np.arcsin(np.sqrt(x[11]) ** 2 / np.sqrt(x[10] ** 2 + x[11] ** 2))
    beta71 = 2 * np.arcsin(np.sqrt(x[13]) ** 2 / np.sqrt(x[12] ** 2 + x[13] ** 2))
    beta81 = 2 * np.arcsin(np.sqrt(x[15]) ** 2 / np.sqrt(x[14] ** 2 + x[15] ** 2))
    
    beta12 = 2 * np.arcsin(np.sqrt(x[2] ** 2 + x[3] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2))
    beta22 = 2 * np.arcsin(np.sqrt(x[6] ** 2 + x[7] ** 2)  / np.sqrt(x[4] ** 2 + x[5] ** 2 + x[6] ** 2 + x[7] ** 2))
    beta32 = 2 * np.arcsin(np.sqrt(x[10] ** 2 + x[11] ** 2) / np.sqrt(x[8] ** 2 + x[9] ** 2 + x[10] ** 2 + x[11] ** 2))
    beta42 = 2 * np.arcsin(np.sqrt(x[10] ** 2 + x[11] ** 2) / np.sqrt(x[12] ** 2 + x[13] ** 2 + x[14] ** 2 + x[15] ** 2))
    
    beta13 = 2 * np.arcsin(np.sqrt(x[4] ** 2 + x[5] ** 2 + x[6] ** 2 + x[7] ** 2)  / np.sqrt(x[0] ** 2 + x[1] ** 2+x[2] ** 2 + x[3] ** 2+x[4] ** 2 + x[5] ** 2+x[6] ** 2 + x[7] ** 2))
    beta23 = 2 * np.arcsin(np.sqrt(x[12] ** 2 + x[13] ** 2 + x[14] ** 2 + x[15] ** 2) / np.sqrt(x[8] ** 2 + x[9] ** 2+x[10] ** 2 + x[11] ** 2+x[12] ** 2 + x[13] ** 2+x[14] ** 2 + x[15] ** 2))
    beta33 = 0

    beta14 = 2 * np.arcsin(
        np.sqrt(x[8] ** 2 + x[9] ** 2+x[10] ** 2 + x[11] ** 2+x[12] ** 2 + x[13] ** 2+x[14] ** 2 + x[15] ** 2) / np.sqrt(x[0] ** 2+ x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2+x[5] ** 2 + x[6] ** 2 + x[7] ** 2 + x[8] ** 2+x[9] ** 2 + x[10] ** 2 + x[11] ** 2 + x[12] ** 2+x[13] ** 2 + x[14] ** 2 + x[15] ** 2)
    )

    return np.array([beta11, beta21, beta31, beta41, beta51, beta61, beta71, beta81, beta12, beta22, beta32, beta42, beta13, beta23, beta33, beta14])
#see this:https://arxiv.org/abs/quant-ph/0407010v1:
#Transformation of quantum states using uniformly controlled rotations

data = np.loadtxt("ee.txt") 
Y= data[:, -1] 
pi = np.pi
X=data[:, 0:10]
pad = 0.001*np.ones((len(X), 6))
X_pad = np.c_[X, pad]

normalization = np.sqrt(np.sum(X_pad ** 2, -1))
X_norm = (X_pad.T / normalization).T

feature = np.array([get_angles(x) for x in X_norm])

#print(X_norm)


################################################################################################################################

plt.figure()
plt.scatter(X[:, 0][Y == 1], X[:, 1][Y == 1], c="r", marker="o", edgecolors="k")
plt.scatter(X[:, 0][Y == 0], X[:, 1][Y == 0], c="b", marker="o", edgecolors="k")
plt.title("Original data")
plt.show()

plt.figure()
dim1 = 0
dim2 = 3
plt.scatter(X_norm[:, dim1][Y == 1], X_norm[:, dim2][Y == 1], c="r", marker="o", edgecolors="k")
plt.scatter(X_norm[:, dim1][Y == 0], X_norm[:, dim2][Y == 0], c="b", marker="o", edgecolors="k")
plt.title("Padded and normalised data".format(dim1, dim2))
plt.show()

plt.figure()
dim1 = 0
dim2 = 15
plt.scatter(feature[:, dim1][Y == 1], feature[:, dim2][Y == 1], c="r", marker="o", edgecolors="k")
plt.scatter(
    feature[:, dim1][Y == 0], feature[:, dim2][Y == 0], c="b", marker="o", edgecolors="k")
plt.title("Feature vectors".format(dim1, dim2))
plt.show()


#######################################################################################################################################

def statepreparation(angel, circuit, wire):
    
   
    circuit.ry(angel[0], wire[0])   
    circuit.cx(wire[0], wire[1])
    circuit.barrier()
    
    circuit.ry(angel[1], wire[1])
    circuit.cx(wire[0], wire[1])
    circuit.barrier()
    
    circuit.ry(angel[2], wire[1])
    circuit.x(wire[0])
    circuit.cx(wire[0], wire[1])
    circuit.barrier()
    
    circuit.ry(angel[3], wire[1])
    circuit.cx(wire[0], wire[1])
    circuit.barrier()    
    
    
    
    circuit.ry(angel[4], wire[1])
    circuit.cx(wire[0], wire[1])
    circuit.barrier()
    
    circuit.ry(angel[5], wire[1])
    circuit.x(wire[0])
    circuit.cx(wire[0], wire[1])
    circuit.barrier()
    
    
    
    circuit.ry(angel[8], wire[2])
    circuit.cx(wire[1], wire[2])
    circuit.barrier()
    
    
    circuit.ry(angel[9], wire[2])
    circuit.cx(wire[1], wire[2])
    circuit.barrier()
    
    circuit.ry(angel[10], wire[2])
    circuit.cx(wire[1], wire[2])
    circuit.barrier()
    
    circuit.ry(angel[11], wire[2])
    circuit.cx(wire[1], wire[2])
    circuit.barrier()
    
    circuit.ry(angel[12], wire[3])
    circuit.cx(wire[2], wire[3])
    circuit.barrier()
    
    circuit.ry(angel[13], wire[3])
    circuit.cx(wire[2], wire[3])
    circuit.barrier()
    
    circuit.ry(angel[14], wire[3])
    circuit.cx(wire[2], wire[3])
    circuit.barrier()
    
    circuit.ry(angel[15], wire[3])
    circuit.cx(wire[2], wire[3])
    circuit.barrier()
    
    return circuit
#######################################################################################################################################

for i in range(0, len(X_norm)):
    x = X_norm[i]
    ang = get_angles(x)
    q       = QuantumRegister(4)
    c       = ClassicalRegister(1)
    circuit = QuantumCircuit(q,c)
    circuit = statepreparation(ang, circuit, [0,1,2,3])
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(circuit, backend=simulator).result()
    statevector = result.get_statevector()


########################################################################################################################################

x = X_norm[0] #for simplification
ang = get_angles(x)
q       = QuantumRegister(4)
c       = ClassicalRegister(1)
circuit = QuantumCircuit(q,c)
circuit = statepreparation(ang, circuit, [0,1,2,3]) #add a for loop here to encode parameters too.
simulator = Aer.get_backend('statevector_simulator')
result = execute(circuit, backend=simulator).result()
statevector = result.get_statevector()
print(statevector)

############################################################################################################################################

circuit.draw('mpl') #state preparation circuit.

##########################################################################################################################################

plot_bloch_multivector(statevector)

#########################################################################################################################################

def u_gate(param, circuit, wire):
    
    circuit.u3(param[0],param[1],param[2],wire)
    return circuit

def cu_gate(param, circuit, control, target):
    
    circuit.cu3(param[0],param[1],param[2], control, target)
    return circuit


def circuit_block(param, circuit, wire):
    
    circuit = u_gate(param[0], circuit, wire[0])
    circuit = u_gate(param[1], circuit, wire[1])
    circuit = u_gate(param[2], circuit, wire[2])
    circuit = u_gate(param[3], circuit, wire[3])
    circuit.cz(wire[0], wire[1])
    circuit.cz(wire[1], wire[2])
    circuit.cz(wire[2], wire[3])
    circuit.cz(wire[0], wire[3])
    circuit.barrier()
    
    
    return circuit

def c_circuit_block(param, circuit, control, target):
    
    circuit = cu_gate(param[0], circuit, control, target[0])
    circuit = cu_gate(param[1], circuit, control, target[1])
    circuit = cu_gate(param[2], circuit, control, target[2])
    circuit = cu_gate(param[3], circuit, control, target[3])
    
    return circuit

def create_circuit(param, circuit, wire):
    
    for i in range(param.shape[0]):
        circuit = circuit_block(param[i], circuit, wire)
        
    return circuit

def create_c_circuit(param, circuit, control, target):
   
    for i in range(param.shape[0]):
        circuit = c_circuit_block(param[i], circuit, control, target)
        
    return circuit


def execute_circuit(params, feature, bias=0, shots=1000):
    
    q       = QuantumRegister(4)
    c       = ClassicalRegister(1)
    circuit = QuantumCircuit(q,c)
    circuit = statepreparation(feature, circuit, [0,1,2,3])

    circuit = create_circuit(params, circuit, [0,1,2,3])
    circuit.measure(0,c)
    result = execute(circuit,backend,shots=shots).result()

    counts = result.get_counts(circuit)
    result=np.zeros(2)
    for key in counts:
        result[int(key,2)]=counts[key]
    result/=shots
    return result[1] + bias

####################################################################################################################################################
#just for example#####

x = X_norm[0]
ang = get_angles(x)

params = np.array([[[pi/6,pi/4,pi/3],
                    [pi/3,pi/4,pi/3],
                   [pi/6,pi/3,pi/6],
                   [pi/4,pi/6,pi/6]],
                   [[pi/3,pi/4,pi/4],
                   [pi/4,pi/4,pi/3],
                   [pi/6,pi/4,pi/6],
                   [pi/4,pi/3,pi/6]],[[pi/3,pi/3,pi/3],
                   [pi/4,pi/4,pi/4],
                   [pi/6,pi/6,pi/6],
                   [pi/4,pi/3,pi/6]],
                  [[pi/3,pi/3,pi/3],
                   [pi/4,pi/4,pi/4],
                   [pi/6,pi/6,pi/6],
                   [pi/4,pi/3,pi/6]]])

q = QuantumRegister(4) # QuantumRegister define number of qubit. here we have 4 qubit.
c = ClassicalRegister(1) # this defines a classical bit for storing measurement results
circuit = QuantumCircuit(q,c) #defines 5 wire, 4 for qubits and 1 for bit. 
circuit = statepreparation(ang, circuit, [0,1,2,3]) 

circuit = create_circuit(params, circuit, [0,1,2,3]) #create_circuit is in qiskit too.

circuit.measure(0, c) # template measure function(qubit=0 or 1, c=classical bit which we want to store measurement results on it)

circuit.draw(output='mpl')

##################################################################################################################################################################

def real(param1, param2, feature, shots=1024):
    
    q = QuantumRegister(5) #one ancilla qubit + 4 qubit.
    c = ClassicalRegister(1)
    circuit = QuantumCircuit(q,c)
    circuit.h(q[0])
    circuit = statepreparation(feature, circuit, [1,2,3,4])
    circuit = create_c_circuit(param1, circuit, 0, [1,2,3,4])
    #entanglement ancilla qubit with feature qubits.
    circuit.cz(q[0], q[1])
    circuit.cz(q[0], q[2])
    circuit.cz(q[0], q[3])
    circuit.cz(q[0], q[4]) 
    circuit.x(q[0])
    circuit = create_c_circuit(param2, circuit, 0, [1,2,3,4])
    circuit.x(q[0])
    circuit.h(q[0])
    circuit.measure(q[0],c)
    result = execute(circuit,backend,shots=shots).result()
    counts = result.get_counts(circuit)
    result=np.zeros(2)
    for key in counts:
        result[int(key,2)]=counts[key]
    result/=shots
    return (2*result[0]-1)

def imaginary(param1, param2, feature, shots=1024):
    
    q = QuantumRegister(5)
    c = ClassicalRegister(1)
    circuit = QuantumCircuit(q,c)
    circuit.h(q[0])
    circuit = statepreparation(feature, circuit, [1,2,3,4])
    circuit = create_c_circuit(param1, circuit, 0, [1,2,3,4])
    circuit.cz(q[0], q[1])
    circuit.cz(q[0], q[2])
    circuit.cz(q[0], q[3])
    circuit.cz(q[0], q[4])
    circuit.x(q[0])
    circuit = create_c_circuit(param2, circuit, 0, [1,2,3,4])
    circuit.x(q[0])
    circuit.u1(np.pi/2, q[0])
    circuit.h(q[0])
    circuit.measure(q[0],c)
    result = execute(circuit,backend,shots=shots).result()
    counts = result.get_counts(circuit)
    result=np.zeros(2)
    for key in counts:
        result[int(key,2)]=counts[key]
    result/=shots
    return -(2*result[0]-1)

def gradients(params, feature, label, bias=0):
    grads = np.zeros_like(params)
    imag = imaginary(params, params, feature)
    for i in range(params.shape[0]):
        for j in range(params.shape[1]):
            params_bis = np.copy(params)
            
            params_bis[i][j][0]+=np.pi
            grads[i][j][0] = -0.5 * real(params, params_bis, feature)
            params_bis[i][j][0]-=np.pi
            
            params_bis[i][j][1]+=np.pi
            grads[i][j][1] = 0.5 * (imaginary(params, params_bis, feature) - imag)
            params_bis[i][j][1]-=np.pi
            
            params_bis[i][j][2]+=np.pi
            grads[i][j][2] = 0.5 * (imaginary(params, params_bis, feature) - imag)
            params_bis[i][j][2]-=np.pi
    p = execute_circuit(params, feature, bias=bias) 
    grad_bias = (p - label) / (p * (1 - p)) 
    grads *= grad_bias
    return grads, grad_bias


#############################################################################################################################################

def predict(probas): #treshold number
    
    return (probas>=0.5)*1


def binary_crossentropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss - l * np.log(p)

    loss = loss / len(labels)
    return loss

def cost(params, features, labels): #params are weights 
    predictions = [execute_circuit(params, f) for f in features]
    return binary_crossentropy(labels, predictions)

def square_loss(labels, predictions): #in supervised learning, the cost function is usually
    
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss 

def accuracy(labels, predictions):

    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss

############################################################################training#######################################################

np.random.seed(0) #With the seed reset (every time), the same set of numbers will appear every time.

#If the random seed is not reset, different numbers appear with every invocation:

num_data = len(Y)
num_train = int(0.8 * num_data)  #x_train, y_train, x_test, y_test = sklearn.model_selection.train_split(x, y, test_size=0.1)
index = np.random.permutation(range(num_data)) #Randomly permute a sequence, or return a permuted range.
feats_train = feature[index[:num_train]]
Y_train = Y[index[:num_train]]
feats_val = feature[index[num_train:]]
Y_val = Y[index[num_train:]]
#print(num_train)
# We need these later for plotting
X_train = X[index[:num_train]]
X_val = X[index[num_train:]]

num_layer = 4

num_qubit = 4

params_init = (0.01 * np.random.randn(4, 4, 3))

#print(params_init)
bias_init = 0.01
batch_size = num_train
learning_rate = 0.008
momentum = 0.4

var = np.copy(params_init)
bias = bias_init
v = np.zeros_like(var) #return an array with the same shape like params_init
v_bias = 0


##########################################################################################################################################

for it in range(200):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, num_train, (batch_size,)) #(lowest, higest, size)
    feats_train_batch = feats_train[batch_index]
    Y_train_batch = Y_train[batch_index]
    grads = np.zeros_like(var)
    grad_bias = 0
    var_corrected = var + momentum * v
    bias_corrected = bias + momentum * v_bias
    
    #with parallel_backend("multiprocessing"): this is for parrallelization. qiskit has also a routine for parallelization. here: https://qiskit.org/documentation/stubs/qiskit.providers.aer.QasmSimulator.html.
        #g, g_bias = zip(*Parallel(n_jobs=-2)(delayed(gradients)(var_corrected, feats_train_batch[j], Y_train_batch[j], bias_corrected) for j in range(batch_size) ))
    for j in range(batch_size):
        g, g_bias = gradients(var_corrected, feats_train_batch[j], Y_train_batch[j], bias_corrected)
        grads += g / batch_size
        grad_bias +=g_bias / batch_size
    v = momentum * v - learning_rate * grads
    v_bias = momentum * v_bias - learning_rate * grad_bias
    
    var += v
    bias += v_bias
    
    #print(var, bias)
    # Compute predictions on train and validation set
    probas_train = np.array([execute_circuit(var, f, bias=bias) for f in feats_train])
    probas_val = np.array([execute_circuit(var, f, bias=bias) for f in feats_val])
    predictions_train = predict(probas_train)
    predictions_val = predict(probas_val)

    # Compute accuracy on train and validation set
    acc_train = accuracy(Y_train, predictions_train)
    acc_val = accuracy(Y_val, predictions_val)

    print("Iter: {:5d} | Loss: {:0.7f} | Acc train: {:0.7f} | Acc validation: {:0.7f} "
       "".format(it + 1, cost(var, feature, Y), acc_train, acc_val))

#################################################end##############################################################################################



