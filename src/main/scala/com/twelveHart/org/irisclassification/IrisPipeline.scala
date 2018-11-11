package com.twelveHart.org.irisclassification


import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.Row
import org.slf4j.LoggerFactory

object IrisPipeline extends IrisWrapper {
  
  val log = LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME)
  
  def main(args: Array[String]): Unit = {
    val vector = "iris-features-column"
    val target = "iris-species-column"
    val outputLabel = "label"
    
    val arr = buildDataFrame(path).toDF(vector, target).randomSplit(Array(0.85, 0.15), 98765L)
    val indexer = new StringIndexer().setInputCol(target).setOutputCol(outputLabel)
    val train = arr(0)
    val test = arr(1)
  
    val classifier: RandomForestClassifier = new RandomForestClassifier()
      .setFeaturesCol(vector)
      .setFeatureSubsetStrategy("sqrt")
      .setNumTrees(15)
      .setMaxDepth(2)
      .setImpurity("gini")
  
    val irisPipeline: Pipeline = new Pipeline().setStages(Array(indexer) ++ Array(classifier))
  
    val grid: Array[ParamMap] = new ParamGridBuilder()
      .addGrid(classifier.numTrees, Array(8, 15, 16, 24, 32, 64, 96))
      .addGrid(classifier.maxDepth, Array(2, 4, 6, 8, 10, 12))
      .addGrid(classifier.impurity, Array("gini", "entropy"))
      .build()
  
    val validated = new TrainValidationSplit()
      .setSeed(1234567L)
      .setEstimatorParamMaps(grid)
      .setEstimator(irisPipeline)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setTrainRatio(0.85)
      .fit(train)
      .transform(test)
  
    validated.show(100)
    
    val prediction = "prediction"
    val results = validated.select(prediction, outputLabel)
    
    val accuracy = new MulticlassClassificationEvaluator()
        .setLabelCol(outputLabel)
        .setMetricName("accuracy")
        .setPredictionCol(prediction)
        .evaluate(results)
    
    println(accuracy)
    
    val metrics = new MulticlassMetrics(
      results.rdd.collect {
        case Row(p: Double, l: Double) => (p, l)
      })
    
    println(s"Accuracy is ${metrics.accuracy} and Weighted Precision is ${metrics.weightedPrecision}")
    session.stop()
  }
}
