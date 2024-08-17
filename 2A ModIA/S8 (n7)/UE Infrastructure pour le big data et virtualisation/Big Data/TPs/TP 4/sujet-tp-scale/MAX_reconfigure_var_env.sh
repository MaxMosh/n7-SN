ssh maxime1@slave1 export HADOOP_HOME=$HOME/bigdata/hadoop-2.7.1
ssh maxime1@slave1 export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

ssh maxime1@slave1 export SPARK_HOME=$HOME/bigdata/spark-2.4.3-bin-hadoop2.7
ssh maxime1@slave1 export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH


ssh maxime2@slave2 export HADOOP_HOME=$HOME/bigdata/hadoop-2.7.1
ssh maxime2@slave2 export PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

ssh maxime2@slave2 export SPARK_HOME=$HOME/bigdata/spark-2.4.3-bin-hadoop2.7
ssh maxime2@slave2 export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
