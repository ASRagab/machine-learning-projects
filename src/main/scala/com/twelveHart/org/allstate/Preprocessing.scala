package com.twelveHart.org.allstate

import com.twelveHart.org.{DataHelper, SparkSessionCreate}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

trait Preprocess {
  implicit val spark: SparkSession
}

class Preprocessing(implicit val spark: SparkSession) extends Preprocess {
  import ColumnHelper._
  
  val trainPath: String = DataHelper.prepareResourceFiles("data", "train.csv.gz")
  val testPath: String = DataHelper.prepareResourceFiles("data", "test.csv.gz")
  
  val trainSrc: DataFrame = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(trainPath)
    .cache()
  
  val testSrc: DataFrame = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(testPath)
    .cache()
  
  private val df = trainSrc
    .withColumnRenamed("loss", "label") //it needs the target column to be called "label"
    .sample(false, 1.0)
  
  val test: Dataset[Row] = testSrc
    .sample(false, 1.0)
    .cache()
  
  private val dropped = df.na.drop()
  private val splits = dropped.randomSplit(Array(0.75, 0.25), 12345L)
  val (train, validation) = (splits(0), splits(1))
  
  private val categoricalColumns = train.columns.withFilter(isCat).withFilter(notTooManyCatsinCol)
  val stageStringIndexer: Array[StringIndexer] = categoricalColumns
    .map(c => new StringIndexer()
      .setInputCol(c)
      .setOutputCol(catNewCol(c)))
  
  private val indexedColumns = categoricalColumns.map(catNewCol)
  val stageOneHotEncoder: OneHotEncoderEstimator = new OneHotEncoderEstimator()
    .setInputCols(indexedColumns)
    .setOutputCols(indexedColumns.map(oneHotCol))
  
  val featureCols: Array[String] = train.columns
    .withFilter(notTooManyCatsinCol)
    .withFilter(isFeatureCol)
    .withFilter(c => !isCat(c))
    .withFilter(c => !isIdx(c))
    .map(identity)
  
  val assembler: VectorAssembler = new VectorAssembler()
    .setInputCols(featureCols)
    .setOutputCol("features")
}


object ColumnHelper {
  def isCat(col: String): Boolean = col.startsWith("cat")
  def isIdx(col: String): Boolean = col.startsWith("idx")
  
  lazy val catNewCol: String => String = col => if (isCat(col)) s"idx_$col" else col
  
  lazy val oneHotCol: String => String = col => if (isIdx(col)) s"ohe_$col" else col
  
  def notTooManyCatsinCol(col: String): Boolean = !(col matches "cat(109$|110$|112$|113$|116$)")
  
  def isFeatureCol(col: String): Boolean = !(col matches "id|label")
}