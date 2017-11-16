package ai.clipper.spark.container.impl

import java.net.UnknownHostException

import ai.clipper.container.data.DoubleVector
import ai.clipper.rpc.RPC
import ai.clipper.spark.{Clipper, SparkModelContainer}
import org.apache.spark.{SparkConf, SparkContext}

object ContainerMain {

  def main(args: Array[String]): Unit = {

    val modelPath = sys.env("CLIPPER_MODEL_PATH")
    val modelName = sys.env("CLIPPER_MODEL_NAME")
    val modelVersion = sys.env("CLIPPER_MODEL_VERSION")

    val clipperAddress = sys.env.getOrElse("CLIPPER_IP", "127.0.0.1")
    val clipperPort = sys.env.getOrElse("CLIPPER_PORT", "7000").toInt

    val conf = new SparkConf()
      .setAppName("ClipperSparkContainer")
      .setMaster("local")
    lazy val sc: SparkContext = {
      val sparkContext = new SparkContext(conf)
      // Reduce logging noise
      sparkContext.parallelize(Seq(""))
        .foreachPartition(x => {
          import org.apache.commons.logging.LogFactory
          import org.apache.log4j.{Level, LogManager}
          LogManager.getRootLogger().setLevel(Level.WARN)
          val log = LogFactory.getLog("EXECUTOR-LOG:")
          log.warn("START EXECUTOR WARN LOG LEVEL")
        })
      sparkContext
    }
    println("Lazy spark context created")
    val container: SparkModelContainer = Clipper.loadSparkModel(sc, modelPath)
    println(s"Loaded spark model: $container")
    val parser = new DoubleVector.Parser

    while (true) {
      println("Starting Clipper Spark Container")
      println(s"Serving model $modelName@$modelVersion")
      println(s"Connecting to Clipper at $clipperAddress:$clipperPort")

      val rpcClient = new RPC(parser)
      try {
        rpcClient.start(container, modelName, modelVersion, clipperAddress, clipperPort)
      } catch {
        case e: UnknownHostException => e.printStackTrace()
      }
    }
  }
}
