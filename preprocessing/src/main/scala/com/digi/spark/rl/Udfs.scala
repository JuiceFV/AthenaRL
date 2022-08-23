package com.digi.spark.rl

import org.apache.spark.sql.functions.coalesce
import org.apache.spark.sql.functions.udf

object Udfs {
    val emptyMap = udf(() => Map.empty[Long, Double])
    val emptyMapOfIds = udf(() => Map.empty[Long, Seq[Long]])
    val emptyMapOfMap = udf(() => Map.empty[Long, Map[Long, Double]])
    val emptyMapOfArrOfMap = udf(() => Map.empty[Long, Seq[Map[Long, Double]]])
    val emptyStr = udf(() => "")
    val emptyArrOfLong = udf(() => Array.empty[Long])
    val emptyArrOfStr = udf(() => Array.empty[String])
    val emptyArrOfDbl = udf(() => Array.empty[Double])
    val emptyArrOfMap = udf(() => Array.empty[Map[Long, Double]])
    val emptyArrOfMapStr = udf(() => Array.empty[Map[String, Double]])
}