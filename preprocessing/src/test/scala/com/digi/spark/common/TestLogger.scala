package com.digi.spark.common.testutil

import org.slf4j.{Logger, LoggerFactory}

trait TestLogger {
    lazy val logger: Logger = LoggerFactory.getLogger(this.getClass.getName)
}
