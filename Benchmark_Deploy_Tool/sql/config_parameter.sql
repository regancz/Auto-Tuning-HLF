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

 Date: 23/12/2023 23:48:16
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for config_parameter
-- ----------------------------
DROP TABLE IF EXISTS `config_parameter`;
CREATE TABLE `config_parameter`  (
  `id` varchar(40) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `Orderer_General_Authentication_TimeWindow` float NULL DEFAULT NULL,
  `Orderer_General_Cluster_SendBufferSize` float NULL DEFAULT NULL,
  `Orderer_General_Keepalive_ServerInterval` float NULL DEFAULT NULL,
  `Orderer_General_Keepalive_ServerMinInterval` float NULL DEFAULT NULL,
  `Orderer_General_Keepalive_ServerTimeout` float NULL DEFAULT NULL,
  `Metrics_Statsd_WriteInterval` float NULL DEFAULT NULL,
  `Orderer_BatchTimeout` float NULL DEFAULT NULL,
  `Orderer_BatchSize_MaxMessageCount` float NULL DEFAULT NULL,
  `Orderer_BatchSize_AbsoluteMaxBytes` float NULL DEFAULT NULL,
  `Orderer_BatchSize_PreferredMaxBytes` float NULL DEFAULT NULL,
  `SmartBFT_RequestBatchMaxCount` float NULL DEFAULT NULL,
  `SmartBFT_RequestBatchMaxInterval` float NULL DEFAULT NULL,
  `SmartBFT_RequestForwardTimeout` float NULL DEFAULT NULL,
  `SmartBFT_RequestComplainTimeout` float NULL DEFAULT NULL,
  `SmartBFT_RequestAutoRemoveTimeout` float NULL DEFAULT NULL,
  `SmartBFT_ViewChangeResendInterval` float NULL DEFAULT NULL,
  `SmartBFT_ViewChangeTimeout` float NULL DEFAULT NULL,
  `SmartBFT_LeaderHeartbeatTimeout` float NULL DEFAULT NULL,
  `SmartBFT_CollectTimeout` float NULL DEFAULT NULL,
  `SmartBFT_IncomingMessageBufferSize` float NULL DEFAULT NULL,
  `SmartBFT_RequestPoolSize` float NULL DEFAULT NULL,
  `SmartBFT_LeaderHeartbeatCount` float NULL DEFAULT NULL,
  `peer_keepalive_minInterval` float NULL DEFAULT NULL,
  `peer_keepalive_client_interval` float NULL DEFAULT NULL,
  `peer_keepalive_client_timeout` float NULL DEFAULT NULL,
  `peer_keepalive_deliveryClient_interval` float NULL DEFAULT NULL,
  `peer_keepalive_deliveryClient_timeout` float NULL DEFAULT NULL,
  `peer_gossip_membershipTrackerInterval` float NULL DEFAULT NULL,
  `peer_gossip_maxBlockCountToStore` float NULL DEFAULT NULL,
  `peer_gossip_maxPropagationBurstLatency` float NULL DEFAULT NULL,
  `peer_gossip_maxPropagationBurstSize` float NULL DEFAULT NULL,
  `peer_gossip_propagateIterations` float NULL DEFAULT NULL,
  `peer_gossip_propagatePeerNum` float NULL DEFAULT NULL,
  `peer_gossip_pullInterval` float NULL DEFAULT NULL,
  `peer_gossip_pullPeerNum` float NULL DEFAULT NULL,
  `peer_gossip_requestStateInfoInterval` float NULL DEFAULT NULL,
  `peer_gossip_publishStateInfoInterval` float NULL DEFAULT NULL,
  `peer_gossip_publishCertPeriod` float NULL DEFAULT NULL,
  `peer_gossip_dialTimeout` float NULL DEFAULT NULL,
  `peer_gossip_connTimeout` float NULL DEFAULT NULL,
  `peer_gossip_recvBuffSize` float NULL DEFAULT NULL,
  `peer_gossip_sendBuffSize` float NULL DEFAULT NULL,
  `peer_gossip_digestWaitTime` float NULL DEFAULT NULL,
  `peer_gossip_requestWaitTime` float NULL DEFAULT NULL,
  `peer_gossip_responseWaitTime` float NULL DEFAULT NULL,
  `peer_gossip_aliveTimeInterval` float NULL DEFAULT NULL,
  `peer_gossip_aliveExpirationTimeout` float NULL DEFAULT NULL,
  `peer_gossip_reconnectInterval` float NULL DEFAULT NULL,
  `peer_gossip_election_startupGracePeriod` float NULL DEFAULT NULL,
  `peer_gossip_election_membershipSampleInterval` float NULL DEFAULT NULL,
  `peer_gossip_election_leaderAliveThreshold` float NULL DEFAULT NULL,
  `peer_gossip_election_leaderElectionDuration` float NULL DEFAULT NULL,
  `peer_gossip_pvtData_pullRetryThreshold` float NULL DEFAULT NULL,
  `peer_gossip_pvtData_transientstoreMaxBlockRetention` float NULL DEFAULT NULL,
  `peer_gossip_pvtData_pushAckTimeout` float NULL DEFAULT NULL,
  `peer_gossip_pvtData_btlPullMargin` float NULL DEFAULT NULL,
  `peer_gossip_pvtData_reconcileBatchSize` float NULL DEFAULT NULL,
  `peer_gossip_pvtData_reconcileSleepInterval` float NULL DEFAULT NULL,
  `peer_gossip_state_checkInterval` float NULL DEFAULT NULL,
  `peer_gossip_state_responseTimeout` float NULL DEFAULT NULL,
  `peer_gossip_state_batchSize` float NULL DEFAULT NULL,
  `peer_gossip_state_blockBufferSize` float NULL DEFAULT NULL,
  `peer_gossip_state_maxRetries` float NULL DEFAULT NULL,
  `peer_authentication_timewindow` float NULL DEFAULT NULL,
  `peer_client_connTimeout` float NULL DEFAULT NULL,
  `peer_deliveryclient_reconnectTotalTimeThreshold` float NULL DEFAULT NULL,
  `peer_deliveryclient_connTimeout` float NULL DEFAULT NULL,
  `peer_deliveryclient_reConnectBackoffThreshold` float NULL DEFAULT NULL,
  `peer_discovery_authCacheMaxSize` float NULL DEFAULT NULL,
  `peer_discovery_authCachePurgeRetentionRatio` float NULL DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = DYNAMIC;

SET FOREIGN_KEY_CHECKS = 1;
