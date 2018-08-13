import numpy as np
import pdb
import seaborn as sns
import random
import matplotlib.pyplot as plt
sns.set(color_codes=True)


external_batch_size = 2400
internal_batchsize = 128


std_dev=100

def generate_state(mean=0, std_dev=std_dev):
	mean =np.random.randint(-100,100)
	result = np.random.normal(mean, std_dev, 280)
	return result

def fake_batch():
	real_entries = min(random.randint(0,internal_batchsize -1),7)
	fake_entries = max(internal_batchsize - real_entries, 3)
	pdb.set_trace
	real_bois = [generate_state() for i in range(real_entries)]
	collapse = [std_dev/fake_entries * i for i in range(fake_entries,0,-1)]
	fake_bois = [generate_state(std_dev=val) for val in collapse]
	output = np.asarray(real_bois + fake_bois)
	return output



def real_batch():
	return np.asarray([generate_state() for i in range(internal_batchsize)])

train_real_batches =  np.asarray([real_batch() for i in range(external_batch_size)])
train_fake_batches =  np.asarray([fake_batch() for i in range(external_batch_size)])


x_train = np.vstack((train_real_batches,train_fake_batches))
y_train = np.asarray([1 for i in train_real_batches] + [0 for i in train_fake_batches])



val_train_batches =  np.asarray([real_batch() for i in range(external_batch_size)])
val_fake_batches =  np.asarray([fake_batch() for i in range(external_batch_size)])
x_test = np.vstack((val_train_batches,val_fake_batches))
y_test = np.asarray([1 for i in val_train_batches] + [0 for i in val_fake_batches])

# fig, axs = plt.subplots(nrows=2, ncols=5)


# fake = real_batch()

# for i, v in enumerate(fake[:5]):
# 	sns.distplot(v, ax=axs[0,i])
# 	axs[0,i].set_xlim([-250, 250])


# for i, v in enumerate(fake[5:]):
# 	sns.distplot(v, ax=axs[1,i])
# 	axs[0,i].set_xlim([-250, 250])

# plt.show(fig)
# pdb.set_trace()