rm -rf /tmp/hadoop*
ssh maxime1@slave1 rm -rf /tmp/hadoop*
ssh maxime2@slave2 rm -rf /tmp/hadoop*

hdfs namenode -format

start-dfs.sh
