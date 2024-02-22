# CopilotFabric

## 主要功能：

- Dashboard指标统计，通过SSH调用Docker指令实现
- 调用优化算法ASPSA，INSGA2对Fabric参数调优
- 性能预测模型，传统机器学习模型和BP神经网络

## 运行

client为前端，Vue、ElementUI

```
npm install
npm run dev
```

nacos，grafana，xxljob需要登录的情况，需要自行启动Nginx进行代理，并修改iframe标签内容

test提供了基本功能测试

server：main.py 启动

