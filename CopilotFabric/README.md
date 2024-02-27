# CopilotFabric

## 主要功能：

- Dashboard指标统计，通过SSH调用Docker指令实现
- 调用优化算法ASPSA，INSGA2对Fabric参数调优
- 性能预测模型，传统机器学习模型和BP神经网络

## 运行

前端为[Auto-Tuning-HLF-Vue](https://github.com/regancz/Auto-Tuning-HLF-Vue)，Vue、ElementUI实现

```
npm install
npm run dev
```

nacos，grafana，xxljob需要登录的情况，需要自行启动Nginx进行代理，并修改iframe标签内容

为了防止CSRF攻击，在跨域的过程中，cookie被过滤掉了，可以考虑开启Nginx通过代理来解决，但目前并不生效，不清楚原因，因此使用的是chrome插件，解决SameSite的限制

（Chrome在很早以前就不支持对SameSite进行显式更改，在Nginx中无论是更改属性还是携带Cookie都无法解决，原因未知。）

test提供了基本功能测试

server：main.py 启动

