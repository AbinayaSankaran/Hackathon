import pyspark as ps
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col, split, udf, sum, max
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor, GeneralizedLinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.classification import XGBoostEstimator
import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars xgboost4j-spark-0.72.jar,xgboost4j-0.72.jar pyspark-shell'
spark = SparkSession.builder.appName("NYCParking").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df_2014 = spark.read.option("header",True).csv("/user/hduser/input/NYC_2014.csv")

required_cols = ["Issue Date", "Violation Code", "Violation Location", "Violation Precinct", 
                "Violation Time", "Violation County", "Violation In Front Of Or Opposite", "Plate Type",
                "Vehicle Color"]

def convert_time_to_hr(time):
        hour = time[0:2]
        try:
                if 'A' in time:
                        hour = int(float(hour))
                elif 'P' in time:
                        hour = int(float(hour)) + 12
                else:
                        hour = int(float(hour))
                return hour
        except ValueError:
                return 0

covert_time_to_hour_udf = udf(lambda t: convert_time_to_hr(t), IntegerType())
split_col = split(df_2014["Issue Date"], '/')
df2 = df_2014.select(required_cols) \
        .na.drop() \
        .withColumn('month', split_col.getItem(0).cast(IntegerType())) \
        .withColumn('date', split_col.getItem(1).cast(IntegerType())) \
        .withColumn('year', split_col.getItem(2).cast(IntegerType())) \
        .withColumn('Violation Hour', covert_time_to_hour_udf(col("Violation Time")))

required_cols.remove('Issue Date')
required_cols.remove('Violation Time')
required_cols.remove('Vehicle Color')
required_cols.remove('Violation Precinct')
required_cols.remove('Plate Type')
required_cols.append('month')
required_cols.append('date')
required_cols.append('Violation Hour')

df3 = df2.groupby(required_cols).count().withColumnRenamed("count", "violation_count")

# Removing the outliers and only predicting 
df_new = df3.filter(col("violation_count") < 30)
df_new.printSchema()
df_new.cache()
print(df_new.count())

(train, test) = df_new.randomSplit([0.8, 0.2])

county_indexer = StringIndexer(inputCol="Violation County", outputCol="violation_county_index")


front_opp_indexer = StringIndexer(inputCol="Violation In Front Of Or Opposite", 
                                outputCol="violation_in_front_opp_index")


input_cols = ["Violation Code", "Violation Location", "Violation Hour", "month", "date", 
                "violation_county_index", "violation_in_front_opp_index"]

feature_cols = [column.lower().replace(" ", "_")+"_vec" for column in input_cols]

one_hot_encoder = OneHotEncoder(inputCols=input_cols, 
                        outputCols=feature_cols)



assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
# LOADING MODEL
lr = LinearRegressionModel(labelCol="violation_count", 
                                featuresCol="features",
                                maxIter=100, 
                                regParam=0.01, 
                                elasticNetParam=0.1)

lr_pipeline = Pipeline(stages=[county_indexer, front_opp_indexer, one_hot_encoder, assembler, lr])
lr_model = lr_pipeline.fit(df_new)

save_model_path = "/models/lr_model"
lr_model.save(save_model_path)
print(f'Model saved at: {save_model_path}')

# XGBOOST
xgboost = XGBoostEstimator(
    featuresCol="features", 
    )

xg_pipe = Pipeline(stages=[county_indexer, front_opp_indexer, one_hot_encoder, assembler, lr])
Xg_model = xg_pipe.fit(df_new)

save_model_path = "/models/xg_model"
Xg_model.save(save_model_path)
print(f'Model saved at: {save_model_path}')

