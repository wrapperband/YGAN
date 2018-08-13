import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, TimeDistributed, Bidirectional, GRU
import pdb
from data_prep import x_train, y_train, x_test, y_test, fake_batch, real_batch


model = Sequential()
model.add(Bidirectional(GRU(200, return_sequences=True),input_shape=(128,280)))
model.add(Dropout(0.5))

model.add(Bidirectional(GRU(200, return_sequences=True)))
model.add(Dropout(0.5))

model.add(Bidirectional(GRU(200)))
model.add(Dropout(0.5))


model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])




result = model.fit(x_train, y_train,
		  validation_split=0.1,
          epochs=10,
          batch_size=100)
score = model.evaluate(x_test, y_test, batch_size=400)
print(model.metrics_names)
print(score)

pdb.set_trace()


# for i in range(10):
# 	print('REAL')
# 	real = np.expand_dims(real_batch(), axis=0)
# 	real_result = model.predict(real)
# 	print(real_result)
# 	if real_result[0][0] < .3:
# 		pdb.set_trace()


# 	print('FAKE')
# 	fake = np.expand_dims(fake_batch(), axis=0)
# 	fake_result = model.predict(fake)
# 	print(fake_result)
# 	if fake_result[0][0] > .7:
# 		pdb.set_trace()

