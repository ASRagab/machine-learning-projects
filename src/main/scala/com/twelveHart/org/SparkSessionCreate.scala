package com.twelveHart.org

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

object SparkSessionCreate {
  def createSession(appName: String, maybeConf: Option[SparkConf] = None): SparkSession = {
    maybeConf match {
      case Some(conf) =>
        SparkSession.builder()
          .config(conf)
          .getOrCreate()
      case None =>
        SparkSession.builder()
          .master("local[*]")
          .config("spark.debug.maxToStringFields", "150")
          .appName(appName)
          .getOrCreate()
    }
  }
}

