import numpy as np
import tensorflow as tf
from skimage.util import view_as_windows

from protopost import ProtoPost
from nd_to_json import nd_to_json, json_to_nd

#TODO: env vars
PORT = 80
DIMENSIONS = 2
LATTICE_SIZE = 64
KERNEL_SIZE = 5

AUTO_REWARD = False #if true, will apply 1.0 reward every step for STDP purposes

LEARNING_RATE = 1e-0
SPIKE_DECAY = 1e-1
POTENTIAL_INCREASE_MIN = 1e-3 #NOTE: same
POTENTIAL_INCREASE_MAX = 1e-3
TRACE_DECAY = 1e-2
REFRACTORY = 0.1
SPIKE_EFFICIENCY = 1.0 #0.9 #prevent transferring 100% of spike, otherwise negative rewards can lead to blocks of always-on neurons; seems unneded if refractory > 0

def normalize(v, axis=-1):
  v = tf.clip_by_value(v, 0.0, 1.0)
  sum = tf.reduce_sum(v, axis, keepdims=True)
  v = v / sum
  return v

#prolly wont work with even sized kernels
def pad_and_slice(x, shape):
  pad = np.floor(np.array(shape) / 2).astype("int32")
  x = tf.pad(x, [[e, e] for e in pad]).numpy()
  x = view_as_windows(x, shape, 1)
  x = x.reshape([-1, np.prod(shape)])
  return x

lattice_shape = [LATTICE_SIZE for _ in range(DIMENSIONS)]
kernel_shape = [KERNEL_SIZE for _ in range(DIMENSIONS)]

#initialization
potential = np.random.uniform(0, 1, lattice_shape).astype("float32")
spikes = np.zeros_like(potential)
weights = np.random.uniform(0, 1, [np.prod(lattice_shape), np.prod(kernel_shape)]).astype("float32")
weights = normalize(weights)
traces = np.zeros_like(weights)

reward = 0

def add_reward(r):
  global reward
  reward += r

def add_potential(data):
  global potential
  potential += json_to_nd(data)

def get_spikes():
  return nd_to_json(spikes.numpy())

def train(reward):
  global potential, spikes, weights, traces

  weights += traces * LEARNING_RATE * reward

  weights = normalize(weights)
  #decay traces
  traces *= (1 - TRACE_DECAY)

def step():
  global potential, spikes, weights, traces, reward

  #automatically increase potential over time
  increase = tf.random.uniform(potential.shape, POTENTIAL_INCREASE_MIN, POTENTIAL_INCREASE_MAX)
  potential += increase

  #where the potential > threshold, spike
  new_spikes = tf.cast(potential >= 1.0, "float32")
  sliced_new_spikes = pad_and_slice(new_spikes, kernel_shape)

  #set potential to 0 where spiked
  potential = tf.where(new_spikes > 0.5, -REFRACTORY, potential)

  #potential to add based on neighboring spikes * weights
  potential_add = sliced_new_spikes * weights
  potential_add *= SPIKE_EFFICIENCY
  potential_add = tf.reshape(tf.reduce_sum(potential_add, axis=-1), lattice_shape)
  potential += potential_add

  #where neighbors are not currently spiking, add trace equal to how recently neighbor spiked
  trace_add = tf.where(sliced_new_spikes==0, pad_and_slice(spikes, kernel_shape), 0)

  #only update traces where we spike
  trace_add *= tf.expand_dims(tf.reshape(new_spikes, [-1]), -1)

  #add to traces
  traces += trace_add
  #TODO: what about the opposite direction (when other neuron spikes, modify our weight?)

  #update fading spikes
  spikes = tf.maximum(new_spikes, spikes)
  spikes *= (1 - SPIKE_DECAY)

  #train
  if AUTO_REWARD:
    reward += 1
  train(reward)

routes = {
    "": lambda _: step(),
    "reward": add_reward,
    "add-potential": add_potential,
    "get-spikes": lambda _: get_spikes()
}

ProtoPost(routes).start(PORT)
