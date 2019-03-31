import logging

import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model



def seperate_data(data):
    input_data = []
    target = []
    for line in data:
        input_data.append(line[0])
        target.append(line[1])
    return(input_data,target)

class LSTMClassificationModel:

    def __init__(self, name, model_path):
        self.filename = model_path+name+".pkl"

    # Create, Train and Test LSTM Classification Model #
    def create_model(self,input_shape,dataset,classes,epochs=500,batch_size=50,divisions=(0.6,0.2,0.2)):
        logging.info("Creating and training a new model at %s with a shape of %s, %s epochs and a batch size of %s. Data size = %s with a division of %s.",self.filename,input_shape,epochs,batch_size,len(dataset),divisions)
        if not dataset:
            raise Exception("No dataset given")
        if not input_shape:
            raise Exception("No input shape given")
        if not classes:
            raise Exception("No classes given")
        
        # dataset is expected to be in the form [ [[input],[output]],... ]
        if len(dataset[0]) != 2:
            raise Exception("Dataset must be in the form [ [[input],[output]],... ] but found elements of length "+str(len(dataset))+ " when it should be 2.")

        # Get percentages of dataset to divide up
        train_percent, validation_percent, test_percent = divisions
        total = train_percent+validation_percent+test_percent
        train_percent = train_percent/total
        validation_percent = validation_percent/total
        test_percent = test_percent/total

        div1 = int(len(dataset)*train_percent)
        div2 = div1 + int(len(dataset)*validation_percent)

        # Randomize the order of the data
        random.shuffle(dataset)

        # Divide up dataset
        train_data = dataset[:div1]
        validation_data = dataset[div1:div2]
        test_data = dataset[div2:]

        train, train_target = seperate_data(train_data)
        validation, validation_target = seperate_data(validation_data)
        test, test_target = seperate_data(test_data)

        train = np.array(train)
        train_target = np.array(train_target)
        validation = np.array(validation)
        validation_target = np.array(validation_target)
        test = np.array(test)
        test_target = np.array(test_target)
        
        # Create Neural Network Model #
        model = Sequential()
        model.add(LSTM(256, input_shape=input_shape))
        model.add(Dense(len(classes), activation='sigmoid'))

        adam = Adam(lr=0.001)
        chk = ModelCheckpoint(self.filename, monitor='val_acc', save_best_only=True, mode='max', verbose=1)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.fit(train, train_target, epochs=epochs, batch_size=batch_size, callbacks=[chk], validation_data=(validation,validation_target))

        #Load the best model and check accuracy on the test data
        model = load_model(self.filename)
        
        predictions = model.predict(test)
        for i in range(len(predictions)):
            maxValue = np.amax(predictions[i])
            for j in range(len(predictions[i])):
                if predictions[i][j] < maxValue:
                    predictions[i][j] = 0
                else:
                    predictions[i][j] = 1

        predictions = predictions.tolist()
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                predictions[i][j] = int(predictions[i][j])

        test_target = test_target.tolist()
        correct = 0
        for i in range(len(test_target)):
            if test_target[i]==predictions[i]:
                correct = correct + 1
            print("Target:",test_target[i],"Prediction:",predictions[i],"Correct:",test_target[i]==predictions[i])
        print("Accuracy: " + str((correct/len(test_target)*100)) + "%")
        self.model = model

    def load_model_file(self):
        self.model = load_model(self.filename)

    def classify(self,data):
        if not self.model:
            self.load_model_file()

        prediction = self.model.predict(np.array([data]))
        maxValue = np.amax(prediction[0])
        for i in range(len(prediction[0])):
            if prediction[0][i] == maxValue:
                print(i)
                return (i,prediction[0][i])


        
