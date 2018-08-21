Text sentiment binary classification by LSTM(Long Short-Term Memory)

1. Database: Data source Mining Tmall user product reviews originate from Tmall.com
2. Data pretreatment: to the mean, normalized and Text segmentation
3. Dropout After adding the Dropout layer, adjust the Dropout parameters to reduce the risk of over-fitting, but the setting of this hyper-parameter requires experience, or how many times to try. But still can not avoid over fitting phenomenon. Keras provides a callback function EarlyStopping (),
4. You can stop training early for Epoch when val_acc decreases. See keras official documentation for keras.callbacks.EarlyStopping (monitor = 'val_loss', min_delta = 0, patience = 0, verbose = 0, mode = 'auto')
(Https://keras.io/callbacks/#earlystopping)
5. Give it a try Often, in engineering practice, we can assume that the model can be stopped early if its performance does not improve in the five consecutive epochs on the test set.
