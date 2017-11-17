package ai.clipper.spark

import java.io.File
import java.util

import ai.clipper.container.ClipperModel
import ai.clipper.container.data.{DataType, DoubleVector, SerializableString}
import ml.combust.bundle.BundleFile
import ml.combust.mleap
import ml.combust.mleap.runtime.transformer.Transformer
import ml.combust.mleap.tensor.DenseTensor
import org.apache.spark.SparkContext
import org.apache.spark.ml.bundle.SparkBundleContext
import org.apache.spark.{ml => sparkml}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, NaiveBayesModel, SVMModel}
import org.apache.spark.mllib.clustering.{BisectingKMeansModel, GaussianMixtureModel, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.regression.{IsotonicRegressionModel, LassoModel, LinearRegressionModel, RidgeRegressionModel}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, GradientBoostedTreesModel, RandomForestModel}
import org.apache.spark.sql.Dataset
import org.json4s._
import org.json4s.jackson.JsonMethods._
import resource.managed

import scala.collection.JavaConversions._
import scala.collection.immutable
import scala.reflect.runtime.universe
import scala.util.Try

abstract class SparkModelContainer extends ClipperModel[DoubleVector] {

  override def getInputType: DataType = {
    DataType.Doubles
  }

  final override def predict(inputVectors: util.ArrayList[DoubleVector]): util.ArrayList[SerializableString] = {
    val inputs = inputVectors.map { x =>
      val doubles = new Array[Double](x.getData.remaining)
      x.getData.get(doubles)
      Vectors.dense(doubles)
    }.toList
    new util.ArrayList(predict(inputs).map(x => new SerializableString(x.toString)))
  }

  def predict(x: List[Vector]): List[Float]

}

abstract class MLlibContainer extends SparkModelContainer {

  def init(sc: SparkContext, model: MLlibModel): this.type

  override def predict(xs: List[Vector]): List[Float]

}

abstract class PipelineModelContainer extends SparkModelContainer {

  def init(sc: SparkContext, model: sparkml.PipelineModel): this.type

  override def predict(xs: List[Vector]): List[Float]

}

case class MLeapModel(transformer: mleap.runtime.transformer.Transformer, bundlePath: String) {

  import mleap.runtime.{LeapFrame, Row, LocalDataset, Dataset}
  import ml.combust.mleap.core.util.VectorConverters
  import org.apache.spark.ml.linalg.Vector

  def predict(rows: Seq[Vector]): Seq[Double] = {
    val frame = LeapFrame(transformer.inputSchema, LocalDataset(rows.map { features =>
      Row(VectorConverters.sparkVectorToMleapTensor(features))
    }))
    transformer.transform(frame).get.dataset.map(_.last.asInstanceOf[Double]).toSeq
  }

  def predict(features: Vector): Double = predict(List(features)).head

  def save(path: String): Unit = {
    new File(path).mkdirs()
    // assume path is directory
    import mleap.runtime.MleapSupport._
    val bundlePath: String = s"jar:file:$path/bundle.zip"
    for (bf <- managed(BundleFile(bundlePath))) {
      transformer.writeBundle.save(bf)
    }
  }
}

object MLeapModel {
  def apply(sparkTransformer: sparkml.Transformer, dataset: Dataset[_]): MLeapModel = {
    import mleap.spark.SparkSupport._
    val path: String = {
      val fileName: String = scala.util.Random.alphanumeric.take(20).mkString
      s"/tmp/$fileName.zip"
    }
    val bundlePath: String = s"jar:file:$path"
    implicit val sbc: SparkBundleContext = SparkBundleContext().withDataset(sparkTransformer.transform(dataset))
    for (bf <- managed(BundleFile(bundlePath))) {
      sparkTransformer.writeBundle.save(bf)
    }
    apply(path)
  }

  def apply(path: String): MLeapModel = {
    val bundlePath: String = s"jar:file:$path${if (path.endsWith(".zip")) "" else "/bundle.zip"}"
    import mleap.runtime.MleapSupport._
    val transformer: Transformer = {
      (for (bf <- managed(BundleFile(bundlePath))) yield {
        bf.loadMleapBundle().get
      }).opt.get.root
    }
    apply(transformer, bundlePath)
  }
}

//case class MLeapNativeModel(transformer: mleap.runtime.transformer.Transformer, path: String) extends MLeapModel
//case class MLeapSparkModel(sparkTransformer: sparkml.Transformer, dataset: Dataset[_])  extends MLeapModel {
//  import mleap.spark.SparkSupport._
//  import mleap.runtime.MleapSupport._
//  import scala.util.Random
//  val path: String = {
//    val fileName: String = Random.alphanumeric.take(20).mkString
//    s"/tmp/$fileName.zip"
//  }
//  val transformer: Transformer = {
//    val bundleFilePath: String = s"jar:file:$path"
//    implicit val sbc: SparkBundleContext = SparkBundleContext().withDataset(sparkTransformer.transform(dataset))
//    for (bf <- managed(BundleFile(bundleFilePath))) {
//      sparkTransformer.writeBundle.save(bf)
//    }
//    (for (bf <- managed(BundleFile(bundleFilePath))) yield {
//      bf.loadMleapBundle().get
//    }).opt.get.root
//  }
//}

class MLeapModelBundleContainer extends SparkModelContainer {
  var model: Option[MLeapModel] = None

  def init(model: MLeapModel): this.type = {
    this.model = Some(model);
    this
  }

  override def predict(xs: List[Vector]): List[Float] = model.get.predict(xs.map(_.asML)).map(_.toFloat).toList
}

sealed abstract class MLlibModel {
  def predict(features: Vector): Double

  def save(sc: SparkContext, path: String): Unit
}

// Classification

// LogisticRegressionModel
case class MLlibLogisticRegressionModel(model: LogisticRegressionModel)
  extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// SVMModel
case class MLlibSVMModel(model: SVMModel) extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// NaiveBayesModel
case class MLlibNaiveBayesModel(model: NaiveBayesModel) extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// Clustering

// BisectingKMeansModel
case class MLlibBisectingKMeansModel(model: BisectingKMeansModel)
  extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// GaussianMixtureModel
case class MLlibGaussianMixtureModel(model: GaussianMixtureModel)
  extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// KMeansModel
case class MLlibKMeansModel(model: KMeansModel) extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// // LDAModel
// case class MLlibLDAModel(model: LDAModel) extends MLlibModel {
//   override def predict(features: Vector): Double  = {
//     model.predict(features)
//   }
//
//   override def save(sc: SparkContext, path: String): Unit = {
//     model.save(sc, path)
//   }
// }
//
// // PowerIterationClusteringModel
// case class MLlibPowerIterationClusteringModel(model: PowerIterationClusteringModel) extends MLlibModel {
//   override def predict(features: Vector): Double  = {
//     model.predict(features)
//   }
//
//   override def save(sc: SparkContext, path: String): Unit = {
//     model.save(sc, path)
//   }
// }
//
// // Features
//
// // ChiSqSelectorModel
// case class MLlibChiSqSelectorModel(model: ChiSqSelectorModel) extends MLlibModel {
//   override def predict(features: Vector): Double  = {
//     model.predict(features)
//   }
//
//   override def save(sc: SparkContext, path: String): Unit = {
//     model.save(sc, path)
//   }
// }
//
// // Word2VecModel
// case class MLlibWord2VecModel(model: Word2VecModel) extends MLlibModel {
//   override def predict(features: Vector): Double  = {
//     model.predict(features)
//   }
//
//   override def save(sc: SparkContext, path: String): Unit = {
//     model.save(sc, path)
//   }
// }
//
// // FPM
//
// // FPGrowthModel
// case class MLlibFPGrowthModel(model: FPGrowthModel) extends MLlibModel {
//   override def predict(features: Vector): Double  = {
//     model.predict(features)
//   }
//
//   override def save(sc: SparkContext, path: String): Unit = {
//     model.save(sc, path)
//   }
// }
//
// // PrefixSpanModel
// case class MLlibPrefixSpanModel(model: PrefixSpanModel) extends MLlibModel {
//   override def predict(features: Vector): Double  = {
//     model.predict(features)
//   }
//
//   override def save(sc: SparkContext, path: String): Unit = {
//     model.save(sc, path)
//   }
// }

//Recommendation

// MatrixFactorizationModel
case class MLlibMatrixFactorizationModel(model: MatrixFactorizationModel)
  extends MLlibModel {
  override def predict(features: Vector): Double = {
    val userId = features(0).toInt
    val productId = features(1).toInt
    model.predict(userId, productId)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// Regression

// IsotonicRegressionModel
case class MLlibIsotonicRegressionModel(model: IsotonicRegressionModel)
  extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features(0))
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// LassoModel
case class MLlibLassoModel(model: LassoModel) extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// LinearRegressionModel
case class MLlibLinearRegressionModel(model: LinearRegressionModel)
  extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// RidgeRegressionModel
case class MLlibRidgeRegressionModel(model: RidgeRegressionModel)
  extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// Tree

// DecisionTreeModel
case class MLlibDecisionTreeModel(model: DecisionTreeModel)
  extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// RandomForestModel
case class MLlibRandomForestModel(model: RandomForestModel)
  extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

// GradientBoostedTreesModel
case class MLlibGradientBoostedTreesModel(model: GradientBoostedTreesModel)
  extends MLlibModel {
  override def predict(features: Vector): Double = {
    model.predict(features)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    model.save(sc, path)
  }
}

object MLlibLoader {
  def metadataPath(path: String): String = s"$path/metadata"

  def getModelClassName(sc: SparkContext, path: String): String = {
    //    val str = sc.textFile(metadataPath(path)).take(1)(0)
    val str = sc.textFile(metadataPath(path)).collect()
    println(s"ALL METADATA:")
    str.foreach(println(_))
    val jsonStr = str.take(1)(0)
    println(s"JSON STRING: $jsonStr")
    val json = parse(jsonStr)
    val JString(className) = (json \ "class")
    // Spark hardcoded the class name for DecisionTreeModel for some reason,
    // then changed the package. We substitute the correct class name.
    if (className == "org.apache.spark.mllib.tree.DecisionTreeModel") {
      "org.apache.spark.mllib.tree.model.DecisionTreeModel"
    } else {
      className
    }
  }

  def load(sc: SparkContext, path: String): MLlibModel = {
    val className = getModelClassName(sc, path)
    // Reflection Code
    val mirror = universe.runtimeMirror(getClass.getClassLoader)
    val modelModule = mirror.staticModule(className)
    val anyInst = mirror.reflectModule(modelModule).instance
    val loader = anyInst.asInstanceOf[org.apache.spark.mllib.util.Loader[_]]
    val model = loader.load(sc, path) match {
      case model: LogisticRegressionModel =>
        MLlibLogisticRegressionModel(model)
      case model: NaiveBayesModel => MLlibNaiveBayesModel(model)
      case model: SVMModel => MLlibSVMModel(model)
      case model: BisectingKMeansModel => MLlibBisectingKMeansModel(model)
      case model: GaussianMixtureModel => MLlibGaussianMixtureModel(model)
      case model: KMeansModel => MLlibKMeansModel(model)
      // case model: LDAModel => MLlibLDAModel(model)
      // case model: PowerIterationClusteringModel => MLlibPowerIterationClusteringModel(model)
      // case model: ChiSqSelectorModel => MLlibChiSqSelectorModel(model)
      // case model: Word2VecModel => MLlibWord2VecModel(model)
      // case model: FPGrowthModel => MLlibFPGrowthModel(model)
      // case model: PrefixSpanModel => MLlibPrefixSpanModel(model)
      case model: MatrixFactorizationModel =>
        MLlibMatrixFactorizationModel(model)
      case model: IsotonicRegressionModel =>
        MLlibIsotonicRegressionModel(model)
      case model: LassoModel => MLlibLassoModel(model)
      case model: LinearRegressionModel => MLlibLinearRegressionModel(model)
      case model: RidgeRegressionModel => MLlibRidgeRegressionModel(model)
      case model: DecisionTreeModel => MLlibDecisionTreeModel(model)
      case model: RandomForestModel => MLlibRandomForestModel(model)
      case model: GradientBoostedTreesModel =>
        MLlibGradientBoostedTreesModel(model)
    }
    model
  }
}
