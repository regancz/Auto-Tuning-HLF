# CopilotFabric

A Fabric-based auto-tuning system that can automatically provide parameter configurations for optimal performance.

## Features

- Web页面实现概览模块，参数调优模块，监控模块
- 调用优化算法SPSA，ASPSA，INSGA2对Fabric参数调优
- 性能预测模型，参数预测性能，支持传统机器学习模型和BP神经网络

## Requirements

```
pip install -r requirements.txt
```

前端：Vue，Element

后端：[Java](https://github.com/regancz/CopilotFabric-Java-Server)，Python，Nginx，Redis，Nacos，Minio，XXL-JOB

Web Database：MySQL，MongoDB（日志）

Blockchain：Caliper，Hyperledger FabricV2.4/2.5，Promethetus，Grafana，Ansible，NFS

所有组件均采用Docker部署，Docker Compose进行容器编排，k8s部署参考repo：https://github.com/GrgChain/baasmanager

## TODO

优化代码结构，模块化设计，参数注入，输出日志，日志存储

## Screenshots

![web_overview](https://github.com/regancz/Auto-Tuning-HLF/blob/master/pic/web_overview.png "web_overview")

![task_log](https://github.com/regancz/Auto-Tuning-HLF/blob/master/pic/task_log.png "task_log")

![nacos_config](https://github.com/regancz/Auto-Tuning-HLF/blob/master/pic/nacos_config.png "nacos_config")

![Dashboard1](https://github.com/regancz/Auto-Tuning-HLF/blob/master/pic/grafana1.png "grafana1")

![Dashboard2](https://github.com/regancz/Auto-Tuning-HLF/blob/master/pic/grafana2.png "grafana2")

![Dashboard3](https://github.com/regancz/Auto-Tuning-HLF/blob/master/pic/grafana3.png "grafana3")

![Dashboard4](https://github.com/regancz/Auto-Tuning-HLF/blob/master/pic/grafana4.png "grafana4")

![query_metric](https://github.com/regancz/Auto-Tuning-HLF/blob/master/pic/query_metric.png "query_metric")





