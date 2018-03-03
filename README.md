# Keras-LSTM-sentiment-classification-master
1.must be done, the database pretreatment, that is, to the mean, normalized and whitening.

2.After replacing reh with tan as an activation function, you have to reduce learningrate by an order of magnitude, otherwise over-fitting occurs.

3.relu should be used in what layer? The answer is that in addition to the last layer of softmax use tanh, other layers can be.

4. On the proportion of Dropout, the paper said to be 0.5, but I found that 0.25 is better, which may need to be adjusted according to the data

5. After adding the Dropout layer, adjust the Dropout parameters to reduce the risk of over-fitting, but the setting of this hyper-parameter requires experience, or how many times to try. But still can not avoid over fitting phenomenon. Keras provides a callback function EarlyStopping (),
You can stop training early for Epoch when val_acc decreases. See keras official documentation for keras.callbacks.EarlyStopping (monitor = 'val_loss', min_delta = 0, patience = 0, verbose = 0, mode = 'auto')
(Https://keras.io/callbacks/#earlystopping)
Give it a try Often, in engineering practice, we can assume that the model can be stopped early if its performance does not improve in the five consecutive epochs on the test set.
