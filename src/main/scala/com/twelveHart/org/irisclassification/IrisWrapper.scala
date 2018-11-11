package com.twelveHart.org.irisclassification

import java.io.File
import java.nio.file.{Files, StandardCopyOption}

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.linalg.Vectors

trait IrisWrapper {
  lazy val session: SparkSession = SparkSession.builder()
    .appName("iris-classification")
    .master("local[*]")
    .getOrCreate()
  
  session.sparkContext.setLogLevel("WARN")
  
  import session.implicits._
  
  val fileName = "/data/iris.csv"
  val path = s"/tmp$fileName"
  val src = Files.copy(
    getClass.getResourceAsStream(fileName),
    new File(path).toPath,
    StandardCopyOption.REPLACE_EXISTING)

  def buildDataFrame(src: String): DataFrame = {
    def res = session.sparkContext.textFile(src)
      .flatMap(l => l.split("\n").toList)
      .map(_.split(","))
      .collect.drop(1)
      .map(row => (Vectors.dense(row(0).toDouble, row(1).toDouble, row(2).toDouble, row(3).toDouble), row(4)))
    
    session.createDataFrame(res)
  }
  
}


