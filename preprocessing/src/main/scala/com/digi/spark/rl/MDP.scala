package com.digi.spark.rl

import scala.math.abs
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.slf4j.LoggerFactory
import org.apache.spark.sql._
import org.apache.spark.sql.functions.coalesce
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

/** Extra columns for the different types of reinforcement learning problems.
  *
  * @param columnName
  *   The name of a custom columns.
  *
  * @param columnType
  *   The column type.
  */
case class ExtraFeatureColumn(columnName: String, columnType: String)

/** Defines a configuration of MDP operator.
  *
  * @param startDsId
  *   The date from which we process the data (ds_id >= startDsId).
  *
  * @param endDsId
  *   The date until which we process the data (ds_id <= endDsId).
  *
  * @param addTerminalRow
  *   Is terminal step required. If true the final row of each episode
  *   corresponds to the last (terminal) step, otherwise it will be omitted.
  *
  * @param inputTableName
  *   The name of original table constructed according to MDP (<S, A, P, R>)
  *   format described below.
  *
  * @param outputTableName
  *   The name of processed table. The new MDP fromat is also described below.
  *
  * @param outlierEpisodeLengthPercentile
  *   The threshold percentile which lies within [0, 1] by one too short/long
  *   episodes are sanitized.
  * @param percentileFunction
  *   Percentile function which used to detect the outliers. `percentile_approx`
  *   by default.
  *
  * @param rewardColumns
  *   The custom names of the reward columns. `metrics` and `reward` by default.
  *   Reward columns consist of two distinct types: 1 - episode reward
  *   (metrics); 2 - step reward (reward).
  *
  * @param extraFeatureColumns
  *   Due to RL is applicable in many others ML fields such as IR, CV, NLP etc.
  *   extra columns may be passed.
  */
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

/** As the input MDP (Markov Decision Process) operator accepts table of the
  * following M<S, A, R, P> format and converts it to the table needed for
  * reinforcement learning MDP.
  *
  * ==Original Table Columns==
  *   - ds_id ( STRING ). A unique ID of given table. Adheres `yyyy-mm-dd` date
  *     format.
  *
  *   - mdp_id ( STRING ). A unique ID of episode. Definition of "episode"
  *     depends on problem. For intance, Ranking Problem defines it as ordered
  *     sequence for given query.
  *
  *   - state_features ( MAP<BIGINT,DOUBLE> ). The features of current step that
  *     are independent on the actions.
  *
  *   - actions ( STRING OR MAP<BIGINT,DOUBLE> ). The action(-s) taken at the
  *     current step. String if the action is discrete or a set of features if
  *     the action is continuous.
  *
  *   - actions_probability ( DOUBLE ). The probability that this(-ese)
  *     action(-s) was taken.
  *
  *   - reward ( DOUBLE ). The reward at the current step.
  *
  *   - metrics ( MAP<STRING,DOUBLE> ). The measure features used to calculate
  *     the reward. I.e. `f(metrics) ~ reward`, the `f()` may be implicitly
  *     defined, but it's crucial that `d(metrics)/dt = d(reward)/dt`.
  *
  *   - sequence_number ( BIGINT ). A number representing the location of the
  *     state in the episode. Note, mdp_id + sequence_number makes unique ID.
  *
  * ==Output Table Columns==
  *   - mdp_id ( STRING ). A unique ID of episode. Definition of "episode"
  *     depends on problem. For intance, Ranking Problem defines it as ordered
  *     sequence for given query.
  *
  *   - state_features ( MAP<BIGINT,DOUBLE> ). The features of current step that
  *     are independent on the actions.
  *
  *   - actions ( STRING OR MAP<BIGINT,DOUBLE> ). The action(-s) taken at the
  *     current step. String if the action is discrete or a set of features if
  *     the action is continuous.
  *
  *   - actions_probability ( DOUBLE ). The probability that this(-ese)
  *     action(-s) was taken.
  *
  *   - reward ( DOUBLE ). The reward at the current step.
  *
  *   - metrics ( MAP<STRING,DOUBLE> ). The measure features used to calculate
  *     the reward. I.e. `f(metrics) ~ reward`, the `f()` may be implicitly
  *     defined, but it's crucial that `d(metrics)/dt = d(reward)/dt`.
  *
  *   - next_state_features ( MAP<BIGINT,DOUBLE> ). The features of the
  *     subsequent step that are actions-independent.
  *
  *   - next_actions (STRING OR MAP<BIGINT, DOUBLE> ). The actions taken at the
  *     next step.
  *
  *   - sequence_number ( BIGINT ). A number representing the location of the
  *     state in the episode before the sequence_number was converted to an
  *     ordinal number. Note, mdp_id + sequence_number makes unique ID.
  *
  *   - time_diff ( BIGINT ). Representing the number of states between the
  *     current state and one of the next n state. If the input table is
  *     sub-sampled states will be missing. This column allows us to know how
  *     many states are missing which can be used to adjust the discount factor.
  *
  *   - time_since_first ( BIGINT ). Representing the number of states between
  *     current state and the very first state of the current episode. If the
  *     input table is sub-sampled states will be missing. This column allows us
  *     to know the derivative (i.e. approach rate) which can be used to adjust
  *     the discount factor.
  *
  * This operator is also suitable for the Reinforcement Learning For Ranking
  * (RL4R) problem. In case RL is applied to the ranking we consider one additional
  * variable - sequence of documents which must be aranged in the optimal way.
  */
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
      Utils.getDataTypes(sqlContext, config.inputTableName, List("actions"))(
        "actions"
      )
    logger.info("Actions column data type:" + s"${actionsDataType}")

    val mdpAdditionalColumns = config.extraFeatureColumns
    var mdpAdditionalColumnDataTypes =
      Utils.getDataTypes(
        sqlContext,
        config.inputTableName,
        mdpAdditionalColumns
      )
    logger.info("MDP additional columns:" + s"${mdpAdditionalColumns}")
    logger.info(
      "MDP additional column types:" + s"${mdpAdditionalColumnDataTypes}"
    )

    val rewardColumnDataTypes =
      Utils.getDataTypes(
        sqlContext,
        config.inputTableName,
        config.rewardColumns
      )
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
      sqlContext
        .sql(s"""
                SELECT mdp_id, COUNT(mdp_id) AS mdp_length
                FROM ${config.inputTableName}
                WHERE ds_id BETWEEN '${config.startDsId}' AND '${config.endDsId}'
                GROUP BY mdp_id
            """)
        .createOrReplaceTempView("episode_length")
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
                ) AS sequence_number_ordinal,
                COALESCE(LEAD(sequence_number) OVER (
                    PARTITION BY
                        mdp_id
                    ORDER BY
                        mdp_id,
                        sequence_number
                ), sequence_number) - sequence_number AS time_diff,
                sequence_number - FIRST(sequence_number) OVER (
                    PARTITION BY
                        mdp_id
                    ORDER BY
                        mdp_id,
                        sequence_number
                ) AS time_since_first
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
        case "string"                         => Udfs.emptyStr()
        case "array<string>"                  => Udfs.emptyArrOfStr()
        case "map<bigint,double>"             => Udfs.emptyMap()
        case "array<map<bigint,double>>"      => Udfs.emptyArrOfMap()
        case "array<bigint>"                  => Udfs.emptyArrOfLong()
        case "map<bigint,array<bigint>>"      => Udfs.emptyMapOfIds()
        case "map<bigint,map<bigint,double>>" => Udfs.emptyMapOfMap()
        case "map<bigint,array<map<bigint,double>>>" =>
          Udfs.emptyMapOfArrOfMap()
      }
      df = df
        .withColumn(nextColName, coalesce(df(nextColName), emptyPlaceholder))
    }

    val stagingTable = "stagingTable_" + config.outputTableName
    if (sqlContext.tableNames.contains(stagingTable)) {
      logger.warn(
        "RL ValidationSql staging table name collision occurred, name: " + stagingTable
      )
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
      case (acc, (k, v)) =>
        s"${acc}, ${k} ${v}, ${Utils.nextStepColumnName(k)} ${v}"
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
            sequence_number_ordinal BIGINT,
            time_diff BIGINT,
            time_since_first BIGINT
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
          case LongType   => res.getAs[Long]("pct")
        }
        val colValCount = res.getAs[Long](s"${columnNamePrefix}_count")
        val outlierCount = res.getAs[Long]("outlier_count")
        logger.info(
          s"Threshold: ${pctVal}; ${columnNamePrefix} count: ${colValCount}; Outlier count: ${outlierCount}"
        )
        val outlierPercent = outlierCount.toDouble / colValCount
        val expectedOutlierPercent = 1.0 - percentile
        if (
          abs(
            outlierPercent - expectedOutlierPercent
          ) / expectedOutlierPercent > 0.1
        ) {
          logger.warn(
            s"Outlier percent mismatch; Expected: ${expectedOutlierPercent}; Got ${outlierPercent}"
          )
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
