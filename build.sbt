// give the user a nice default project!

lazy val root = (project in file(".")).

  settings(
    inThisBuild(List(
      organization := "com.twelveHart.org",
      scalaVersion := "2.12.7"
    )),
    name := "machine-learning-projects",
    version := "0.0.1",
    sparkVersion := "2.4.0",
    sparkComponents := Seq(),

    javacOptions ++= Seq("-source", "1.8", "-target", "1.8"),
    javaOptions ++= Seq("-Xms512M", "-Xmx6g", "-XX:+CMSClassUnloadingEnabled", "-XX:ReservedCodeCacheSize=512M"),
    scalacOptions ++= Seq("-deprecation", "-unchecked"),
    parallelExecution in Test := false,
    fork in run := true,

    coverageHighlighting := true,

    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "2.4.0" % "provided",
      "org.apache.spark" %% "spark-streaming" % "2.4.0" % "provided",
      "org.apache.spark" %% "spark-sql" % "2.4.0" % "provided",
      "org.apache.spark" %% "spark-mllib" % "2.4.0" % "provided",

      "org.scalanlp" %% "breeze" % "0.13.2",
      "org.scalanlp" %% "breeze-natives" % "0.13.2",
      "org.scalanlp" %% "breeze-viz" % "0.13.2",

      "joda-time" % "joda-time" % "2.10.1",
      "ch.qos.logback" % "logback-classic" % "1.2.3",
      "org.scalatest" %% "scalatest" % "3.0.5" % "test",
      "org.scalacheck" %% "scalacheck" % "1.13.4" % "test"
    ),

    // uses compile classpath for the run task, including "provided" jar (cf http://stackoverflow.com/a/21803413/3827)
    run in Compile := Defaults.runTask(fullClasspath in Compile, mainClass in (Compile, run), runner in (Compile, run)).evaluated,

    scalacOptions ++= Seq("-deprecation", "-unchecked"),
    pomIncludeRepository := { _ => false },

   resolvers ++= Seq(
      "sonatype-releases" at "https://oss.sonatype.org/content/repositories/releases/",
      "Typesafe repository" at "http://repo.typesafe.com/typesafe/releases/",
      "Second Typesafe repo" at "http://repo.typesafe.com/typesafe/maven-releases/",
      Resolver.sonatypeRepo("public")
    ),

    pomIncludeRepository := { x => false },

    // publish settings
    publishTo := {
      val nexus = "https://oss.sonatype.org/"
      if (isSnapshot.value)
        Some("snapshots" at nexus + "content/repositories/snapshots")
      else
        Some("releases"  at nexus + "service/local/staging/deploy/maven2")
    }
  )
