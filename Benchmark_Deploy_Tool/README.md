# Benchmark-Deploy-Tool

## 主要功能：

- 一键部署Hyperledger Fabric
- 指定参数及步长进行Benchmark
- 解析Benchmark结果进行持久化，MySQL

## 运行

sql文件夹建表

修改 param_range.yaml其中lower，upper，step上下限和步长，注意单位

修改config.yaml，服务器，数据库，参数配置地址

main.py 启动
