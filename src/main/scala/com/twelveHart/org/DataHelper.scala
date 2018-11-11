package com.twelveHart.org

import java.io.File
import java.nio.file.{Files, StandardCopyOption}

object DataHelper {
  def prepareResourceFiles(subDir: String, fileName: String, path: Option[String] = None): String = {
    val tempPath = path.getOrElse(s"/tmp/$fileName")
    Files.copy(
      getClass.getResourceAsStream(s"/$subDir/$fileName"),
      new File(tempPath).toPath,
      StandardCopyOption.REPLACE_EXISTING)
    
    tempPath
  }
}
