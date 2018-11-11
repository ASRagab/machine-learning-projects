package com.twelveHart.org.allstate

import com.twelveHart.org.{DataHelper, SparkSessionCreate}

object ExploratoryDA {
  def main(args: Array[String]): Unit = {
    val spark = SparkSessionCreate.createSession("exploratory-da")
    
    val path = DataHelper.prepareResourceFiles("data", "train.csv.gz")
    
    val data = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(path)
      .cache()
    
    val df = data.withColumnRenamed("loss", "label")
    df.createOrReplaceTempView("insurance")
    
    spark.sql(
      """SELECT AVG(insurance.label) AS avg_loss,
        |       MIN(insurance.label) AS min_loss,
        |       MAX(insurance.label) AS max_loss
        |FROM insurance
      """.stripMargin).show()
    
    spark.stop()
  }
}
