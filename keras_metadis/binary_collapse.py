import numpy as np
import pdb
import random

external_batch_size = 1200
internal_batchsize = 100

def fake_batch():
	real_entries = random.randint(0,internal_batchsize)
	fake_entries = internal_batchsize - real_entries
	collapsed_boi = np.random.randint(1000, size=(10))
	real_bois = [np.random.randint(1000, size=(10)) for i in range(real_entries)]
	fake_bois = [collapsed_boi for boi in range(fake_entries)]
	output = np.asarray(real_bois + fake_bois)
	return output

def real_batch():
	return np.asarray([np.random.randint(1000, size=(10)) for i in range(internal_batchsize)])

train_real_batches =  np.asarray([real_batch() for i in range(external_batch_size)])
train_fake_batches =  np.asarray([fake_batch() for i in range(external_batch_size)])
x_train = np.vstack((train_real_batches,train_fake_batches))
y_train = np.asarray([1 for i in train_real_batches] + [0 for i in train_fake_batches])


val_train_batches =  np.asarray([real_batch() for i in range(external_batch_size)])
val_fake_batches =  np.asarray([fake_batch() for i in range(external_batch_size)])
x_test = np.vstack((val_train_batches,val_fake_batches))
y_test = np.asarray([1 for i in val_train_batches] + [0 for i in val_fake_batches])

# pdb.set_trace()