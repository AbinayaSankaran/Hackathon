from kafka import KafkaProducer                                                                                         
from random import randint                                                                                              
from time import sleep                                                                                                  
import sys
import pandas as pd
import pickle
                                                                                                                        
BROKER = "broker:2182"                                                                                    
TOPIC = 'NYC'                                                                                                                                                                                 
                                                                                                                        
try:                                                                                                                    
    p = KafkaProducer(bootstrap_servers=BROKER)                                                                         
except Exception as e:                                                                                                  
    print(f"ERROR --> {e}")                                                                                             
    sys.exit(1)

#Loading the model and the labelencoders from saved pickle file

le1 = pickle.load(open("lr_model.pkl", "rb"))[1]
le2 = pickle.load(open("xg_model.pkl", "rb"))[2]


#Loading the data and pre-processing(feature engineering)
df = pd.read_csv('NYC_2014.csv')

df = df.drop(['Issue Date', 'Violation Time', 'Vehicle Color', 'Violation Precinct', 'Plate Type', 'month','date','Violation hours'])

cols = df.columns.values.tolist()

for col in cols:
  df = df[(df[col].notnull())]

df = df[(df['Vehicle Year']!= 0)]
df['Plate ID'] = df['Plate ID'].map(str)
df['Registration State'] = df['Registration State'].map(str)
df['Sub Division'] = df['Sub Division'].map(str)

X,y = df.drop('Violation Location',axis=1), df['Violation Location']


X['Plate ID'] = le1.transform(X['Plate ID'])
X['Registration State'] = le1.transform(X['Registration State'])
X['Sub Division'] = le2.transform(X['Sub Division'])



data = X.values.tolist()


while True:                                                                                                             
    for row in data:
        message = ''
        for val in row:
            message += str(val) + ' '                                                                                        
        print(f">>> '{message.split()}'")                                                                                           
        p.send(TOPIC, bytes(message, encoding="utf8"))                                                                      
        sleep(randint(1,4))