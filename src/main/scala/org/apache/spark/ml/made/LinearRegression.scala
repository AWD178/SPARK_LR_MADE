package org.apache.spark.ml.made


import breeze.linalg.{sum, DenseVector => BreezeDenseVector}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasMaxIter, HasStepSize, HasTol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model, PredictorParams}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}


trait LinearRegressionParams extends PredictorParams
  with HasMaxIter with HasTol with HasStepSize {
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  setDefault(maxIter -> 100, stepSize -> 1e-3, tol -> 1e-8)

  protected def validateAndTransformSchema(schema: StructType): StructType =
    super.validateAndTransformSchema(schema, fitting = true, featuresDataType = new VectorUDT())
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel]
  with LinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  def setStepSize(value: Double): this.type = set(stepSize, value)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()
    val datasetExt: DataFrame = dataset.withColumn("bias_col", lit(1))
    val assembler: VectorAssembler = new VectorAssembler()
      .setInputCols(Array($(featuresCol), "bias_col", $(labelCol)))
      .setOutputCol("features_all")
    val vectors: Dataset[Vector] = assembler.transform(datasetExt).select("features_all").as[Vector]
    val numFeatures: Int = vectors.first.size - 1

    var w: BreezeDenseVector[Double] = BreezeDenseVector.rand[Double](numFeatures)

    for (_ <- 0 until $(maxIter)) {
      val summary = vectors.rdd
        .mapPartitions((data: Iterator[Vector]) => {
          val summarizer = new MultivariateOnlineSummarizer()
          data.foreach(vector => {
            val X = vector.asBreeze(0 until w.size)
            val y = vector.asBreeze(w.size)
            val res = sum(X * w) - y
            val gradient = X *:* res
            summarizer.add(org.apache.spark.mllib.linalg.Vectors.fromBreeze(gradient))
          })
          Iterator(summarizer)
        })
        .reduce(_ merge _)
      w = w - $(stepSize) * summary.mean.asBreeze
    }

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(w(0 until w.size - 1)).toDense, w(-1)))
      .setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val weights: DenseVector,
                                           val bias: Double
                                         ) extends
  Model[LinearRegressionModel]
  with LinearRegressionParams
  with MLWritable {

  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("LinearRegressionModel"), weights, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(weights, bias), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf =
      dataset.sqlContext.udf.register("transform_" + uid,
        (x: Vector) => {
          weights.dot(x) + bias
        }
      )

    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))

  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      sqlContext.createDataFrame(Seq(Tuple1(weights.asInstanceOf[Vector]))).write.parquet(path + "/vectors")
    }
  }
}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()
      implicit val double_encoder : Encoder[Double] = ExpressionEncoder()

      val (weights, bias) =  vectors.select(vectors("_1").as[Vector], vectors("_2").as[Double]).first()

      val model = new LinearRegressionModel(weights.toDense, bias)
      metadata.getAndSetParams(model)
      model
    }
  }
}