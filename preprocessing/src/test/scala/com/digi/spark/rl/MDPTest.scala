package com.digi.spark.rl

import java.io.File
import org.scalactic.TolerantNumerics
import org.scalatest.Assertions._

import com.digi.spark.common.testutil.PiplineTester

class MDPTest extends PiplineTester {
    implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(0.01)

    test("filter-outliers") {
        val sqlCtx = sqlContext
        import sqlCtx.implicits._
        val sparkContext = sqlCtx.sparkContext

        val percentileFunc = "percentile"

        val config = MDPConfiguration(
            "2022-08-23",
            "2022-08-23",
            false,
            "some_rl_input_1",
            "some_rl_mdp_1",
            Some(0.95),
            percentileFunc
        )

        Utils.dropTrainingTable(sqlContext, s"${config.outputTableName}")
        
        val rl_input = sparkContext
            .parallelize(
                for (mdp_id <- 1 to 100; mdp_length = if (mdp_id <= 95) 2 else 100; sequence_id <- 1 to mdp_length)
                    yield
                        if (mdp_id == 1 && sequence_id == 1)
                            (
                                "2022-08-23",
                                "mdp_1",
                                1,
                                1.0,
                                "item1",
                                0.8,
                                Map(1L -> 1.0),
                                Map("NDCG@1" -> 1.0)
                            )
                        else
                            (
                                "2022-08-23",
                                s"mdp_${mdp_id}",
                                sequence_id,
                                0.228,
                                s"item${(sequence_id + 1) % 2 + 1}",
                                0.7,
                                Map(2L -> 1.0),
                                Map("NDCG@1" -> 3.0)
                            )

            )
            .toDF(
                "ds_id",
                "mdp_id",
                "sequence_number",
                "reward",
                "actions",
                "actions_probability",
                "state_features",
                "metrics"
            )
        rl_input.createOrReplaceTempView(config.inputTableName)

        MDP.run(sqlContext, config)

        assert(
            Utils.outputTableIsValid(
                sqlContext,
                s"${config.outputTableName}",
                "string",
                Map("reward" -> "double", "metrics" -> "map<string,double>")
            )
        )

        val df = 
            sqlCtx.sql(s"""SELECT ${Constants.RL_DATA_COLUMN_NAMES
            .mkString(",")} FROM ${config.outputTableName}""")

        df.show()
        assert(df.count() == 95)
    }

    test("three-states-ranking-mdp") {
        val sqlCtx = sqlContext
        import sqlCtx.implicits._
        val sparkContext = sqlCtx.sparkContext

        val config = MDPConfiguration(
            "2022-08-23",
            "2022-08-23",
            true,
            "some_rl4r_input_2",
            "some_rl4r_mdp_2",
            rewardColumns = List("slate_reward", "item_reward"),
            extraFeatureColumns = List("state_sequence_features")
        )

        val state_sequence_features = Map(
            666L -> List(
                Map(228L -> 0.3, 222L -> 0.5),
                Map(228L -> 0.3, 222L -> 0.5),
                Map(228L -> 0.3, 222L -> 0.5),
                Map(228L -> 0.3, 222L -> 0.5)
            )
        )
        
        Utils.dropTrainingTable(sqlContext, s"${config.outputTableName}")

        val rl4r_input = sparkContext
            .parallelize(
                List(
                    (
                        "2022-08-23",
                        "mdp_1",
                        1,
                        Map(123L -> 0.1, 234L -> 0.2),
                        Map(123001L -> Map(0L -> 0.1, 1L -> 0.2), 234001L -> Map(0L -> 0.2, 3L -> 0.4)),
                        List(0L, 1L, 2L, 3L),
                        0.8,
                        Map(555L -> 1.0),
                        state_sequence_features
                    ),
                    (
                        "2022-08-23",
                        "mdp_1",
                        2,
                        Map(123L -> 0.3, 234L -> 0.2),
                        Map(123001L -> Map(0L -> 0.1, 1L -> 0.2), 234001L -> Map(0L -> 0.2, 3L -> 0.4)),
                        List(3L, 1L, 2L, 0L),
                        0.2,
                        Map(555L -> 2.0),
                        state_sequence_features
                    ),
                    (
                        "2022-08-23",
                        "mdp_1",
                        3,
                        Map(123L -> 0.1, 234L -> 0.3),
                        Map(123001L -> Map(0L -> 0.1, 1L -> 0.1), 234001L -> Map(0L -> 0.4, 3L -> 0.4)),
                        List(3L, 2L, 1L, 0L),
                        0.5,
                        Map(555L -> 3.0),
                        state_sequence_features
                    )
                )
            )
            .toDF(
                "ds_id",
                "mdp_id",
                "sequence_number",
                "slate_reward",
                "item_reward",
                "actions",
                "actions_probability",
                "state_features",
                "state_sequence_features"
            )
        rl4r_input.createOrReplaceTempView(config.inputTableName)

        MDP.run(sqlContext, config)

        assert(Utils.outputTableIsValid(
            sqlContext,
            s"${config.outputTableName}",
            "array<bigint>",
            Map(
                "slate_reward" -> "map<bigint,double>",
                "item_reward" -> "map<bigint,map<bigint,double>>"
            ),
            Map(
                "state_sequence_features" -> "map<bigint,array<map<bigint,double>>>"
            )
        ))

        val df = 
            sqlCtx
                .sql(s"""SELECT ${Constants.RL4R_DATA_COLUMN_NAMES
                    .mkString(",")} FROM ${config.outputTableName}""")
                .sort($"sequence_number".asc)

        df.show(false)
        assert(df.count() == 3)
        val firstRow = df.head
        assert(firstRow.getAs[String](0) == "2022-08-23")
        assert(firstRow.getAs[String](1) == "mdp_1")
        assert(firstRow.getAs[Long](2) == 1L)
        assert(firstRow.getAs[Map[Long, Double]](3) == Map(123L -> 0.1, 234L -> 0.2))
        assert(
            firstRow.getAs[Map[Long, Map[Long, Double]]](4) == Map(
                123001L -> Map(0L -> 0.1, 1L -> 0.2),
                234001L -> Map(0L -> 0.2, 3L -> 0.4)
            )
        )
        assert(firstRow.getAs[List[Long]](5) === List(0L, 1L, 2L, 3L))
        assert(firstRow.getAs[Double](6) == 0.8)
        assert(firstRow.getAs[Map[Long, Double]](7) == Map(555L -> 1.0))
        assert(firstRow.getAs[Map[Long, List[Map[Long, Double]]]](8) == state_sequence_features)
        assert(firstRow.getAs[List[Long]](9) === List(3L, 1L, 2L, 0L))
        assert(firstRow.getAs[Map[Long, Double]](10) == Map(555L -> 2.0))
        assert(firstRow.getAs[Map[Long, List[Map[Long, Double]]]](11) == state_sequence_features)

        val secondRow = df.collect()(1)
        assert(secondRow.getAs[String](0) == "2022-08-23")
        assert(secondRow.getAs[String](1) == "mdp_1")
        assert(secondRow.getAs[Long](2) == 2L)
        assert(secondRow.getAs[Map[Long, Double]](3) == Map(123L -> 0.3, 234L -> 0.2))
        assert(
            secondRow.getAs[Map[Long, Map[Long, Double]]](4) == Map(
                123001L -> Map(0L -> 0.1, 1L -> 0.2),
                234001L -> Map(0L -> 0.2, 3L -> 0.4)
            )
        )
        assert(secondRow.getAs[List[Long]](5) === List(3L, 1L, 2L, 0L))
        assert(secondRow.getAs[Double](6) == 0.2)
        assert(secondRow.getAs[Map[Long, Double]](7) == Map(555L -> 2.0))
        assert(secondRow.getAs[Map[Long, List[Map[Long, Double]]]](8) == state_sequence_features)
        assert(secondRow.getAs[List[Long]](9) === List(3L, 2L, 1L, 0L))
        assert(secondRow.getAs[Map[Long, Double]](10) == Map(555L -> 3.0))
        assert(secondRow.getAs[Map[Long, List[Map[Long, Double]]]](11) == state_sequence_features)
        
        val thirdRow = df.collect()(2)
        assert(thirdRow.getAs[String](0) == "2022-08-23")
        assert(thirdRow.getAs[String](1) == "mdp_1")
        assert(thirdRow.getAs[Long](2) == 3L)
        assert(thirdRow.getAs[Map[Long, Double]](3) == Map(123L -> 0.1, 234L -> 0.3))
        assert(
            thirdRow.getAs[Map[Long, Map[Long, Double]]](4) == Map(
                123001L -> Map(0L -> 0.1, 1L -> 0.1),
                234001L -> Map(0L -> 0.4, 3L -> 0.4)
            )
        )
        assert(thirdRow.getAs[List[Long]](5) === List(3L, 2L, 1L, 0L))
        assert(thirdRow.getAs[Double](6) == 0.5)
        assert(thirdRow.getAs[Map[Long, Double]](7) == Map(555L -> 3.0))
        assert(thirdRow.getAs[Map[Long, List[Map[Long, Double]]]](8) == state_sequence_features)
        assert(thirdRow.getAs[List[Long]](9) === List())
        assert(thirdRow.getAs[Map[Long, Double]](10) == Map())
        assert(thirdRow.getAs[Map[Long, List[Map[Long, Double]]]](11) == Map())
    }
}