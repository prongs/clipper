from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession, Row

from cifar_utils import *
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.pyspark import deploy_pyspark_model

if __name__ == '__main__':
    from pyspark.ml.util import _jvm

    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    cifar_loc = "/Users/rajat.khandelwal/Git/clipper/examples/tutorial"
    train_x, train_y = filter_data(
        *load_cifar(cifar_loc, cifar_filename="cifar_train.data", norm=True))
    # test_x, test_y = filter_data(
    #     *load_cifar(cifar_loc, cifar_filename="cifar_test.data", norm=True))
    df = sc.parallelize(Row(label=float(y), features=Vectors.dense(list(float(i) for i in x))) for (y, x) in
                        zip(train_y, train_x)).toDF()
    lr = LogisticRegression(maxIter=5, regParam=0.01, elasticNetParam=0.5)
    predCol = lr.getPredictionCol()
    pipeline = Pipeline(stages=[lr]).fit(df)
    model = pipeline
    version = os.environ.get("CLIPPER_MODEL_VERSION", "blah")
    _jvm().ai.clipper.spark.Clipper.deploySparkModel(sc._jsc.sc(), "lr-mleap", version,
                                                     _jvm().ai.clipper.spark.MLeapModel.apply(model._to_java(),
                                                                                              df._jdf),
                                                     _jvm().py4j.reflection.ReflectionUtil.classForName(
                                                         "ai.clipper.spark.MLeapModelBundleContainer"),
                                                     "localhost", _jvm().scala.collection.immutable.List.empty(),
                                                     _jvm().scala.Option.apply(None), _jvm().scala.Option.apply(None),
                                                     True)

    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.connect()


    def predict(spark, model, inputs):
        df = spark.sparkContext.parallelize(Row(features=Vectors.dense(list(float(i) for i in x))) for x in
                                            inputs).toDF()
        return [x[predCol] for x in model.transform(df).select(predCol).collect()]


    deploy_pyspark_model(clipper_conn, "lr-pyspark", version, "doubles", predict, model, sc)
