/*
 Navicat Premium Data Transfer

 Source Server         : localhost_3306
 Source Server Type    : MySQL
 Source Server Version : 80030
 Source Host           : localhost:3306
 Source Schema         : auto-tuning-hlf

 Target Server Type    : MySQL
 Target Server Version : 80030
 File Encoding         : 65001

 Date: 23/12/2023 23:48:23
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for performance_metric
-- ----------------------------
DROP TABLE IF EXISTS `performance_metric`;
CREATE TABLE `performance_metric`  (
  `id` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `succ` float NULL DEFAULT NULL,
  `fail` float NULL DEFAULT NULL,
  `send_rate` float NULL DEFAULT NULL,
  `max_latency` float NULL DEFAULT NULL,
  `min_latency` float NULL DEFAULT NULL,
  `avg_latency` float NULL DEFAULT NULL,
  `throughput` float NULL DEFAULT NULL,
  `bench_config` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL,
  `config_id` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
