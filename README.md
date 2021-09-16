# Hackathon 2 - Team sluggers
Problem Statement : NYC Parking Tickets - Violation Location Prediction
---
Members:
Name:-
Abinaya 
Nagender Yadav. 

Parking Violation predictor

In the city of New York, The NYC Department of Finance collects data on every parking ticket issued in NYC (~10M per year!). This data is made publicly available to aid in ticket resolution and to guide policymakers. Some inspirations include:

When are tickets most likely to be issued? Any seasonality?
Where are tickets most commonly issued?
What are the most common years and types of cars to be ticketed?
This project is about training a model on big data and predicting Violation locations using New York parking violations data. 

Build the ML Model using Pyspark. Choose the right algorithms to predict violation locations. Data pre-processing is needed. The application should be modelled using Spark, and the same can be streamed to the Kafka server. Apply containerisation principles as a better software engineering practice. With the help of spark streaming, predict streamed data.   

The model can be deployed in a Kubernetes environment if you consider scalability and self-healing ability are important. The choice of Kubernetes can be your own infrastructure or on any of the popular cloud environments such as Google Kubernetes Engine (GKE) for example.    

Dataset: New York parking violations 

Keywords: Data pre-processing, regression, random forest, XGBoost, Kafka, Spark, Containers

#### Dataset link: https://www.kaggle.com/new-york-city/nyc-parking-tickets?select=Parking_Violations_Issued_-_Fiscal_Year_2016.csv
---
### Service description
Docker container with a seperate container for : 
1. kafka-zookeeper
2. kafka-broker
3. kafka-streamer
4. hadoop_hive
5. spark

### Command to start server
`docker-compose up -d`

---

