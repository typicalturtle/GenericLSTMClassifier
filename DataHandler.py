import logging

import csv
import os

class DataHandler:

    def __init__(self, classes, data_path):
        self.classes = classes
        self.file_path = data_path

    def load_data(self,window_size,step=1):
        logging.info("Loading data from %s", self.file_path)
        data_lost = 0
        
        data = []
        
        # Load data of each classification
        for i in range(len(self.classes)):
            
            # Generate target
            target = []
            for j in range(len(self.classes)):
                if i==j:
                    target.append(1)
                else:
                    target.append(0)

            # Load input data file
            for file in os.listdir(self.file_path+self.classes[i]):
                with open(self.file_path+self.classes[i]+"/"+file, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    input_data_string = list(reader)
                    input_data = []
                    
                    # Convert to float values
                    for line in input_data_string:
                        new_line = []
                        for s in line:
                            new_line.append(float(s))
                        input_data.append(new_line)

                    # Split into multiple chunks of the window size
                    length = len(input_data) - window_size
                    if length < 0:
                        logging.warn("Cannot use file %s as it has a length of %s which is less than the window size of %s",file,len(input_data),window_size)
                        data_lost = data_lost + len(input_data)
                    else:
                        for k in range(0,length+1, step):
                            if window_size+k <= len(input_data):
                                data.append([input_data[k:window_size+k],target])
                            else:
                                data_lost = data_lost + step
        logging.info("Data loaded with a loss of %s",data_lost)
        return data
            
        
