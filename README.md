# CopilotFabric

前端采用Vue，Element，后端采用Java，Python编写，Rpc采用Thrift，算法和后端数据库为MySQL，日志数据库采用MongoDB，配置中心采用Nacos，对象存储采用Minio，基准测试采用Caliper，定时任务采用XXL-JOB，监控采用Promethetus和Grafana。自动部署采用自动化运维工具Ansible和分布式文件共享的网络协议NFS。反向代理采用Nginx，任务执行支持Python/Bash，缓存采用Redis，区块链采用Hyperledger \ FabricV2.4/2.5，所有组件均采用Docker部署，Docker Compose进行容器编排。

## 主要功能

- 看板，概览模块，任务模块，监控模块和日志模块
- 调用优化算法SPSA，ASPSA，MOASPSA，MOPSO对Fabric参数调优
- 性能预测模型，传统机器学习模型和BP神经网络

## 运行

```
conda create --name CopilotFabric python=3.8
source activate CopilotFabric
or
conda activate CopilotFabric
pip install -r requirements.txt
```
