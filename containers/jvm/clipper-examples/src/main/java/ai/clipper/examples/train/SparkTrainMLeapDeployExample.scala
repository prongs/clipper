package ai.clipper.examples.train

import ai.clipper.spark._
import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object SparkTrainMLeapDeployExample {

  /**
    *
    * This example can be run with the following spark-submit command when run with
    * Spark 2.1.
    * 1. Define the following environment variables:
    * + CLIPPER_MODEL_NAME=<name>
    * + CLIPPER_MODEL_VERSION<version>
    * + CLIPPER_HOST=<host>
    * + SSH_USER=<user> # only needed if CLIPPER_HOST isn't localhost
    * + SSH_KEY_PATH=<key_path> # only needed if CLIPPER_HOST isn't localhost
    * + SPARK_HOME=<path-to-spark>
    * + CLIPPER_HOME=<path-to-clipper>
    * 2. Build the application:
    * + `cd $CLIPPER_HOME/containers/jvm && mvn clean package`
    *
    * 3. Run with Spark
    * + `$SPARK_HOME/bin/spark-submit --master "local[2]" --class ai.clipper.examples.train.SparkDFTrainDeployExample --name <spark-app-name> \
    * $CLIPPER_HOME/containers/jvm/clipper-examples/target/clipper-examples-0.1.jar`
    *
    */
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("ClipperTest").master("local[*]").getOrCreate()
    val sparkHome = sys.env("SPARK_HOME")
    // Load and parse the data file.
    val training: DataFrame = spark.read.format("libsvm").load(s"$sparkHome/data/mllib/sample_libsvm_data.txt")
    // Split the data into training and test sets (30% held out for testing)
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Fit the model
    val lrModel: LogisticRegressionModel = lr.fit(training)
    // Print the coefficients and intercept for logistic regression
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")


    val clipperHost = sys.env.getOrElse("CLIPPER_HOST", "localhost")
    val clipperVersion = sys.env.getOrElse("CLIPPER_MODEL_VERSION", "1")
    val sshKeyPath = sys.env.get("SSH_KEY_PATH")
    val sshUser = sys.env.get("SSH_USER")

    Clipper.deploySparkModel(spark.sparkContext,
      sys.env("CLIPPER_MODEL_NAME"),
      clipperVersion,
      MLeapModel(lrModel, training),
      classOf[MLeapModelBundleContainer],
      clipperHost,
      List("a"),
      sshUser,
      sshKeyPath)
  }
}
