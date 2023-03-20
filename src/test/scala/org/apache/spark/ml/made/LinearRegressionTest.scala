package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import com.google.common.io.Files
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers._


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta: Double = 0.01
  val weights: DenseVector[Double] = LinearRegressionTest._weights
  val bias: Double = LinearRegressionTest._bias
  val y_true: DenseVector[Double] = LinearRegressionTest._y
  val df: DataFrame = LinearRegressionTest._df

  private def validateModel(data: DataFrame): Unit = {
    val y_pred = data.collect().map(_.getAs[Double](1))

    y_pred.length should be (100000)

    for (i <- y_pred.indices) {
      y_pred(i) should be (y_true(i) +- delta)
    }
  }

  private def validateEstimator(model: LinearRegressionModel) = {
    model.weights.size should be (weights.size)
    model.weights(0) should be (weights(0) +- delta)
    model.weights(1) should be (weights(1) +- delta)
    model.weights(2) should be (weights(2) +- delta)
    model.bias should be (bias +- delta)
  }

  "Estimator" should "create working model" in {
    val estimator = new LinearRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setMaxIter(1000)
      .setStepSize(0.1)

    val model = estimator.fit(df)

    validateEstimator(model)
  }

  "Model" should "make right predictions of target variable" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.fromBreeze(weights).toDense,
      bias = bias
    ).setLabelCol("label")
      .setPredictionCol("prediction")
      .setFeaturesCol("features")

    validateModel(model.transform(df))
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setMaxIter(100)
        .setStepSize(1.0)
    ))

    val tempDir = Files.createTempDir()

    pipeline.write.overwrite().save(tempDir.getAbsolutePath)

    val model = Pipeline.load(tempDir.getAbsolutePath)
      .fit(df)
      .stages(0)
      .asInstanceOf[LinearRegressionModel]

    validateEstimator(model)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setLabelCol("label")
        .setFeaturesCol("features")
        .setPredictionCol("prediction")
        .setMaxIter(100)
        .setStepSize(1.0)
    ))

    val model = pipeline.fit(df)

    val tempDir = Files.createTempDir()

    print(tempDir.getAbsolutePath)

    model.write.overwrite().save(tempDir.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(tempDir.getAbsolutePath)

    validateModel(reRead.transform(df))
  }
}

object LinearRegressionTest extends WithSpark {
  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand[Double](100000, 3)
  lazy val _weights: DenseVector[Double] = DenseVector(2.1, 0.4, -0.5)
  lazy val _bias: Double = 1.0
  lazy val _y: DenseVector[Double] = _X * _weights + _bias + DenseVector.rand(100000) * 0.0001
  lazy val _df: DataFrame = createDataFrame(_X, _y)

  def createDataFrame(X: DenseMatrix[Double], y: DenseVector[Double]): DataFrame = {
    import sqlc.implicits._

    lazy val data: DenseMatrix[Double] = DenseMatrix.horzcat(X, y.asDenseMatrix.t)

    lazy val df = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq
      .toDF("x1", "x2", "x3", "label")

    lazy val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3"))
      .setOutputCol("features")

    lazy val _df: DataFrame = assembler
      .transform(df)
      .select("features", "label")

    _df
  }
}