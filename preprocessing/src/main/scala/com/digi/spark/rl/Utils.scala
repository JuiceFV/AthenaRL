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

    def dropTrainingTable(
        sqlContext: SQLContext,
        tableName: String
    ): Unit = 
        try {
            val dropTableQuery = s"DROP TABLE ${tableName}"
            sqlContext.sql(dropTableQuery)
        } catch {
            case e: org.apache.spark.sql.catalyst.analysis.NoSuchTableException => {}
            case e: Throwable                                                   => logger.error(e.toString())
        }
    
    def outputTableIsValid(
        sqlContext: SQLContext,
        tableName: String,
        actionsDataType: String = "string",
        rewardTypes: Map[String, String] = Constants.DEFAULT_REWARD_TYPES,
        mdpAdditionalTypes: Map[String, String] = Map(),
        isArray: Boolean = false
    ): Boolean = {
        val dt = sqlContext.sparkSession.catalog
            .listColumns(tableName)
            .collect
            .map(column => column.name -> column.dataType)
            .toMap

        val nextActiosnDataType = this.nextStepColumnType(actionsDataType, isArray)
        (
            actionsDataType == dt.getOrElse("actions", "") &&
            nextActiosnDataType == dt.getOrElse("next_actions", "") &&
            rewardTypes.filter { case (k, v) => (v == dt.getOrElse(k, "")) }.size == rewardTypes.size &&
            mdpAdditionalTypes.filter {
                case (k, v) => 
                    (v == dt.getOrElse(k, "") &&
                        this.nextStepColumnType(v, isArray) == dt.getOrElse(
                            this.nextStepColumnName(k),
                            ""
                        )
                    )
            }.size == mdpAdditionalTypes.size
        )
    }
}