package com.digi.spark.rl

import org.slf4j.LoggerFactory
import org.apache.spark.sql._

object Utils {
    private val logger = LoggerFactory.getLogger(this.getClass.getName)

    def nextStepColumnName(columnName: String): String = 
        "next_" + columnName

    def nextStepColumnType(columnType: String, isArray: Boolean): String =
        if (isArray) s"array<${columnType}>" else columnType

    def getDataTypes(
        sqlContext: SQLContext, 
        tableName: String, 
        columnNames: List[String]
    ): Map[String, String] = {
        val dataTypes = sqlContext.sparkSession.catalog
            .listColumns(tableName)
            .collect
            .filter(column => columnNames.contains(column.name))
            .map(column => column.name -> column.dataType)
            .toMap
        dataTypes
    }

}