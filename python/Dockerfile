FROM ubuntu:18.04
LABEL team="Hydra"

WORKDIR /app

ENV PYTHONUNBUFFERED 1
ENV DISPLAY=:99

SHELL ["/bin/bash", "-c"]

# Install apt dependencies
RUN apt update

# Install dependencies
RUN apt install wget -y
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN apt install openjdk-8-jdk -y
RUN apt install scala -y
RUN pip3 install pyspark==2.4.6
RUN pip3 install nltk
RUN pip3 install numpy
RUN pip3 install pandas

# Installing XGBoost
RUN apt install cmake -y
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install ez_setup
RUN pip3 install wheel
RUN pip3 install sklearn
RUN pip3 install xgboost

# Downloading Spark
RUN wget https://mirrors.estointernet.in/apache/spark/spark-3.1.2/spark-3.1.2-bin-hadoop2.7.tgz
RUN tar -xvf spark-3.1.2-bin-hadoop2.7.tgz
RUN mv spark-3.1.2-bin-hadoop2.7 spark
RUN rm spark-3.1.2-bin-hadoop2.7.tgz

# Downloading Spark Streaming Kafka
RUN wget https://search.maven.org/remotecontent?filepath=org/apache/spark/spark-streaming-kafka-0-8-assembly_2.11/2.4.8/spark-streaming-kafka-0-8-assembly_2.11-2.4.8.jar -O spark-streaming-kafka-0-8-assembly_2.11-2.4.8.jar

# Backup original bashrc
RUN cp ~/.bashrc ~/.bashrc.bak

# adding spark_config script
COPY spark_config.sh /app/spark_config.sh
RUN chmod u+x /app/spark_config.sh
RUN /app/spark_config.sh

ENV PYSPARK_PYTHON python3

# Application code
COPY transformer.py /app/transformer.py
COPY XGBoost_model.pkl /app/XGBoost_model.pkl
COPY start_spark.sh /app/start_spark.sh
RUN chmod u+x /app/start_spark.sh

CMD ["/bin/bash","-c","/app/start_spark.sh"]


# Downloading kafka
RUN wget https://archive.apache.org/dist/kafka/2.2.0/kafka_2.12-2.2.0.tgz
RUN tar -xvf kafka_2.12-2.2.0.tgz
RUN mv kafka_2.12-2.2.0 kafka
RUN rm kafka_2.12-2.2.0.tgz

# Install python-kafka
RUN pip3 install kafka-python

# Downloading hadoop
RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-2.8.5/hadoop-2.8.5.tar.gz
RUN tar -xvf hadoop-2.8.5.tar.gz
RUN mv hadoop-2.8.5 hadoop
RUN rm hadoop-2.8.5.tar.gz

# Backup original bashrc and hadoop-env.sh
RUN cp ~/.bashrc ~/.bashrc.bak
RUN cp /app/hadoop/etc/hadoop/hadoop-env.sh /app/hadoop/etc/hadoop/hadoop-env.sh_bak

# adding hadoop_config script
COPY hadoop_config.sh /app/hadoop_config.sh
RUN chmod u+x /app/hadoop_config.sh
RUN /app/hadoop_config.sh

# Adding SSH authentication
RUN ssh-keygen -q -t rsa -N '' -f ~/.ssh/id_rsa
RUN cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# Allow SSH local connections
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config

# update bashrc
RUN source ~/.bashrc

# Format HDFS file System
RUN /app/hadoop/bin/hdfs namenode -format

# Downloading Hive
RUN wget http://archive.apache.org/dist/hive/hive-2.3.5/apache-hive-2.3.5-bin.tar.gz
RUN tar -xvf apache-hive-2.3.5-bin.tar.gz
RUN mv apache-hive-2.3.5-bin hive
RUN rm apache-hive-2.3.5-bin.tar.gz

# Backup original bashrc and hive-env.sh
RUN cp ~/.bashrc ~/.bashrc.bak

# adding hive_config script
COPY hive_config.sh /app/hive_config.sh
RUN chmod u+x /app/hive_config.sh
RUN /app/hive_config.sh

CMD source ~/.bashrc && /etc/init.d/ssh start && /app/hadoop/sbin/start-dfs.sh && /app/hive/bin/hive --service metastore




