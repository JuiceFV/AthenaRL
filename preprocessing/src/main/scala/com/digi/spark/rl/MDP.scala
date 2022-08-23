package com.digi.spark.rl

import scala.math.abs
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.slf4j.LoggerFactory
import org.apache.spark.sql._
import org.apache.spark.sql.functions.coalesce
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

case class ExtraFeatureColumn(columnName: String, columnType: String)

case class MDPConfiguration(
    startDsId: String,
    endDsId: String,
    addTerminalRow: Boolean,
    inputTableName: String,
    outputTableName: String,
    outlierEpisodeLengthPercentile: Option[Double] = None,
    percentileFunction: String = "percentile_approx",
    rewardColumns: List[String] = Constants.DEFAULT_REWARD_COLUMNS,
    extraFeatureColumns: List[String] = Constants.DEFAULT_EXTRA_FEATURE_COLUMNS
)

object MDP {
    private val logger = LoggerFactory.getLogger(this.getClass.getName);
    def run(
        sqlContext: SQLContext, 
        config: MDPConfiguration
    ): Unit = {
        var filterTerminal = "WHERE next_state_features IS NOT NULL"
        if (config.addTerminalRow) {
            filterTerminal = "";
        }
        val actionsDataType = 
            Utils.getDataTypes(sqlContext, config.inputTableName, List("actions"))("actions")
        logger.info("Actions column data type:" + s"${actionsDataType}")

        val mdpAdditionalColumns = config.extraFeatureColumns
        var mdpAdditionalColumnDataTypes = 
            Utils.getDataTypes(sqlContext, config.inputTableName, mdpAdditionalColumns)
        logger.info("MDP additional columns:" + s"${mdpAdditionalColumns}")
        logger.info("MDP additional column types:" + s"${mdpAdditionalColumnDataTypes}")

        val rewardColumnDataTypes = 
            Utils.getDataTypes(sqlContext, config.inputTableName, config.rewardColumns)
        logger.info("reward columns:" + s"${config.rewardColumns}")
        logger.info("reward column types:" + s"${rewardColumnDataTypes}")

        MDP.createTrainingTable(
            sqlContext,
            config.outputTableName,
            actionsDataType,
            rewardColumnDataTypes,
            mdpAdditionalColumnDataTypes
        )

        config.outlierEpisodeLengthPercentile.foreach { percentile =>
            sqlContext.sql(s"""
                SELECT mdp_id, COUNT(mdp_id) AS mdp_length
                FROM ${config.inputTableName}
                WHERE ds_id BETWEEN '${config.startDsId}' AND '${config.endDsId}'
                GROUP BY mdp_id
            """).createOrReplaceTempView("episode_length")
        }

        val mdpLengthThreshold = MDP.lengthThreshold(
            sqlContext,
            config.outlierEpisodeLengthPercentile,
            "mdp",
            config.percentileFunction,
            "episode_length"
        )

        val mdpFilter = mdpLengthThreshold
            .map { threshold => 
                s"""mdp_filter AS (
                    SELECT mdp_id 
                    FROM episode_length 
                    WHERE mdp_length <= ${threshold}
                ),"""
            }
            .getOrElse("")

        val joinClause = mdpLengthThreshold
            .map { threshold => 
                s"""
                JOIN mdp_filter
                WHERE a.mdp_id = mdp_filter.mdp_id AND
                """.stripMargin
            }
            .getOrElse("WHERE")

        val rewardSourceColumns = rewardColumnDataTypes.foldLeft("") {
            case (acc, (k, v)) => s"${acc}, a.${k}"
        }

        val mdpAdditionalSourceColumns = mdpAdditionalColumnDataTypes.foldLeft("") {
            case (acc, (k, v)) => s"${acc}, a.${k}"
        }

        val sourceTable = s"""
            WITH ${mdpFilter}
                source_table AS (
                    SELECT
                        a.mdp_id,
                        a.state_features,
                        a.actions_probability,
                        a.actions
                        ${rewardSourceColumns},
                        a.sequence_number
                        ${mdpAdditionalSourceColumns}
                    FROM ${config.inputTableName} a
                    ${joinClause}
                    a.ds_id BETWEEN '${config.startDsId}' AND '${config.endDsId}'
                )
        """.stripMargin

        val rewardColumnsQuery = rewardColumnDataTypes.foldLeft("") {
            case (acc, (k, v)) => s"${acc}, ${k}"
        }

        val mdpAdditionalColumnsQuery = mdpAdditionalColumnDataTypes.foldLeft("") {
            case (acc, (k, v)) =>
                s"""
                ${acc},
                ${k},
                LEAD(${k}) OVER (
                    PARTITION BY
                        mdp_id
                    ORDER BY
                        mdp_id,
                        sequence_number
                ) AS ${Utils.nextStepColumnName(k)}
                """
        }

        
        val sqlCommand = s"""
        ${sourceTable},
        joined_table AS (
            SELECT
                mdp_id,
                state_features,
                actions,
                LEAD(actions) OVER (
                    PARTITION BY
                        mdp_id
                    ORDER BY
                        mdp_id,
                        sequence_number
                ) AS next_actions,
                actions_probability
                ${rewardColumnsQuery},
                LEAD(state_features) OVER (
                    PARTITION BY
                        mdp_id
                    ORDER BY
                        mdp_id,
                        sequence_number
                ) AS next_state_features,
                sequence_number,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        mdp_id
                    ORDER BY
                        mdp_id,
                        sequence_number
                ) AS sequence_number_ordinal
                ${mdpAdditionalColumnsQuery}
            FROM source_table
            CLUSTER BY HASH(mdp_id, sequence_number)
        )
        SELECT * FROM joined_table
        ${filterTerminal}
        """.stripMargin

        logger.info("Executing query: ")
        logger.info(sqlCommand)
        var df = sqlContext.sql(sqlCommand)
        logger.info("Done with query")

        val handleCols = mdpAdditionalColumnDataTypes.++(
            Map(
                "actions" -> actionsDataType,
                "state_features" -> "map<bigint,double>"
            )
        )

        
        for ((colName, colType) <- handleCols) {
            val nextColName = Utils.nextStepColumnName(colName)
            val emptyPlaceholder = colType match {
                case "string"                                => Udfs.emptyStr()
                case "array<string>"                         => Udfs.emptyArrOfStr()
                case "map<bigint,double>"                    => Udfs.emptyMap()
                case "array<map<bigint,double>>"             => Udfs.emptyArrOfMap()
                case "array<bigint>"                         => Udfs.emptyArrOfLong()
                case "map<bigint,array<bigint>>"             => Udfs.emptyMapOfIds()
                case "map<bigint,map<bigint,double>>"        => Udfs.emptyMapOfMap()
                case "map<bigint,array<map<bigint,double>>>" => Udfs.emptyMapOfArrOfMap()
            }
            df = df
                .withColumn(nextColName, coalesce(df(nextColName), emptyPlaceholder))
        }

        val stagingTable = "stagingTable_" + config.outputTableName
        if (sqlContext.tableNames.contains(stagingTable)) {
            logger.warn("RL ValidationSql staging table name collision occurred, name: " + stagingTable)
        }
        df.createOrReplaceTempView(stagingTable)

        val insertCommandOutput = s"""
        INSERT OVERWRITE TABLE ${config.outputTableName} PARTITION(ds_id='${config.endDsId}')
        SELECT * FROM ${stagingTable}
        """.stripMargin
        sqlContext.sql(insertCommandOutput)
    }

    def createTrainingTable(
        sqlContext: SQLContext,
        tableName: String,
        actionsDataType: String,
        rewardColumnDataTypes: Map[String, String] = Map("reward" -> "double"),
        mdpAdditionalColumnDataTypes: Map[String, String] = Map()
    ): Unit = {
        val rewardColumns = rewardColumnDataTypes.foldLeft("") {
            case (acc, (k, v)) => s"${acc}, ${k} ${v}"
        }
        
        val mdpAdditionalColumns = mdpAdditionalColumnDataTypes.foldLeft("") {
            case (acc, (k, v)) => s"${acc}, ${k} ${v}, ${Utils.nextStepColumnName(k)} ${v}"            
        }

        val sqlQuery = s"""
        CREATE TABLE IF NOT EXISTS ${tableName} (
            mdp_id STRING,
            state_features MAP<BIGINT, DOUBLE>,
            actions ${actionsDataType},
            next_actions ${actionsDataType},
            actions_probability DOUBLE
            ${rewardColumns},
            next_state_features MAP<BIGINT, DOUBLE>,
            sequence_number BIGINT,
            sequence_number_ordinal BIGINT
            ${mdpAdditionalColumns}
        ) PARTITIONED BY (ds_id STRING) TBLPROPERTIES ('RETENTION'='30')
        """.stripMargin
        sqlContext.sql(sqlQuery);
    }

    def lengthThreshold(
        sqlContext: SQLContext, 
        columnPercentile: Option[Double], 
        columnNamePrefix: String,
        percentile_function: String,
        tempTableName: String
    ): Option[Double] = 
        columnPercentile.flatMap { percentile =>
            {
                val df = sqlContext.sql(s"""
                    WITH a AS (
                        SELECT ${percentile_function}(${columnNamePrefix}_length, ${percentile}) pct FROM ${tempTableName}
                    ),
                    b AS (
                        SELECT
                            count(*) as ${columnNamePrefix}_count,
                            sum(IF(${tempTableName}.${columnNamePrefix}_length > a.pct, 1, 0)) as outlier_count
                        FROM ${tempTableName} CROSS JOIN a
                    )
                    SELECT a.pct, b.${columnNamePrefix}_count, b.outlier_count
                    FROM b CROSS JOIN a
                """)
                val res = df.first
                val pctVal = res.schema("pct").dataType match {
                    case DoubleType => res.getAs[Double]("pct")
                    case LongType => res.getAs[Long]("pct")
                }
                val colValCount = res.getAs[Long](s"${columnNamePrefix}_count")
                val outlierCount = res.getAs[Long]("outlier_count")
                logger.info(s"Threshold: ${pctVal}; ${columnNamePrefix} count: ${colValCount}; Outlier count: ${outlierCount}")
                val outlierPercent = outlierCount.toDouble / colValCount
                val expectedOutlierPercent = 1.0 - percentile
                if (abs(outlierPercent - expectedOutlierPercent) / expectedOutlierPercent > 0.1) {
                    logger.warn(s"Outlier percent mismatch; Expected: ${expectedOutlierPercent}; Got ${outlierPercent}")
                    None
                } else 
                    Some(pctVal)
            }

        }

    def main(configJson: String): Unit = {
        val sparkSession = SparkSession.builder().enableHiveSupport().getOrCreate()
        val mapper = new ObjectMapper()
        mapper.registerModule(DefaultScalaModule)
        val mdpConfig = mapper.readValue(configJson, classOf[MDPConfiguration])
        MDP.run(sparkSession.sqlContext, mdpConfig)
        sparkSession.stop()
    }

}