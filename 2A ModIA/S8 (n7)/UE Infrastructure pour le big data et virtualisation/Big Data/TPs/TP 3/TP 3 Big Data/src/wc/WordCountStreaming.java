package wc;

import java.util.Arrays;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaPairDStream;
import org.apache.spark.streaming.api.java.JavaReceiverInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;

public class WordCountStreaming { 

	public static void main(String args[]) {

		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);

		// Create the context with a 1 second batch size 
		SparkConf sparkConf = new SparkConf().setAppName("WordCountStreaming"); 
		JavaStreamingContext ssc = new JavaStreamingContext(sparkConf, Durations.seconds(1)); 

		// Create a JavaReceiverInputDStream on target ip:port and count the 
		// words in input stream of \n delimited text (eg. generated by 'nc') 
		JavaReceiverInputDStream<String> lines = ssc.socketTextStream("localhost", 9999); 
		JavaDStream<String> words = lines.flatMap(x -> Arrays.asList(x.split(" ")).iterator()); 
		JavaPairDStream<String, Integer> wordCounts = words.mapToPair(s -> new Tuple2<>(s, 1)) 
				.reduceByKey((i1, i2) -> i1 + i2); 

		wordCounts.print(); 

		ssc.start(); 
		try { 
			ssc.awaitTermination(); 
		} catch (InterruptedException e) { 
			e.printStackTrace(); 
		} 
	} 
}
