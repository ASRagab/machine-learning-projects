package com.twelveHart.org.allstate

import com.twelveHart.org.SparkSessionCreate
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql._

object LR {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSessionCreate.createSession("lr")
    val preprocessing = new Preprocessing
    
    import spark.implicits._
    
    val folds = 3
    val iters = Seq(10)
    val regParam = Seq(0.001)
    val tol = Seq(1e-6)
    val elasticNet = Seq(0.001)
    
    val model = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("label")
    
    println("Building Pipeline")
    val pipeline = new Pipeline()
      .setStages(
        preprocessing.stageStringIndexer :+
        preprocessing.stageOneHotEncoder :+
        preprocessing.assembler :+
        model)
    
    val paramGridBuilder = new ParamGridBuilder()
      .addGrid(model.maxIter, iters)
      .addGrid(model.regParam, regParam)
      .addGrid(model.tol, tol)
      .addGrid(model.elasticNetParam, elasticNet)
      .build()
    
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGridBuilder)
      .setNumFolds(folds)
    
    println("Fitting cross-validation model")
    val modelCV = cv.fit(preprocessing.train)
    
    println("")
    val predictionsLabelsTrain = modelCV.transform(preprocessing.train)
      .select("label", "prediction")
      .map {
        case Row(l: Double, p: Double) => (l, p)
      }.rdd
    
    val predictionLabelsValidation = modelCV.transform(preprocessing.validation)
      .select("label", "prediction")
      .map {
        case Row(l: Double, p: Double) => (l, p)
      }.rdd
    
    
    println("Computing Regression Metrics")
    val metricsTrain = new RegressionMetrics(predictionsLabelsTrain)
    val metricsValidation = new RegressionMetrics(predictionLabelsValidation)
    
    val best = modelCV.bestModel.asInstanceOf[PipelineModel]
    
    val results = "\n=====================================================================\n" +
      s"Param trainSample: 1.0\n" +
      s"Param testSample: 1.0\n" +
      s"TrainingData count: ${preprocessing.train.count}\n" +
      s"ValidationData count: ${preprocessing.validation.count}\n" +
      s"TestData count: ${preprocessing.test.count}\n" +
      "=====================================================================\n" +
      s"Param maxIter = ${iters.mkString(",")}\n" +
      s"Param numFolds = $folds\n" +
      "=====================================================================\n" +
      s"Training data MSE = ${metricsTrain.meanSquaredError}\n" +
      s"Training data RMSE = ${metricsTrain.rootMeanSquaredError}\n" +
      s"Training data R-squared = ${metricsTrain.r2}\n" +
      s"Training data MAE = ${metricsTrain.meanAbsoluteError}\n" +
      s"Training data Explained variance = ${metricsTrain.explainedVariance}\n" +
      "=====================================================================\n" +
      s"Validation data MSE = ${metricsValidation.meanSquaredError}\n" +
      s"Validation data RMSE = ${metricsValidation.rootMeanSquaredError}\n" +
      s"Validation data R-squared = ${metricsValidation.r2}\n" +
      s"Validation data MAE = ${metricsValidation.meanAbsoluteError}\n" +
      s"Validation data Explained variance = ${metricsValidation.explainedVariance}\n" +
      s"CV params explained: ${modelCV.explainParams}\n" +
      s"LR params explained: ${best.stages.last.asInstanceOf[LinearRegressionModel].explainParams}\n" +
      "=====================================================================\n"
    
    println(results)
    
    val output = "output/result_LR.csv"
    println("Run and save prediction of test set")
    modelCV.transform(preprocessing.test)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "loss")
      .coalesce(1)
      .write
      .option("header", "true")
      .format("com.databricks.spark.csv")
      .mode(SaveMode.Overwrite)
      .save(output)
    
    spark.stop()
  }
}
