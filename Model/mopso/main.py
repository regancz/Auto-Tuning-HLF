# encoding: utf-8
import yaml

from Mopso import *
import p_objective


def main():
    particals = 40  # 粒子群的数量
    cycle_ = 20  # 迭代次数
    mesh_div = 10  # 网格等分数量
    thresh = 300  # 外部存档阀值

    # Problem = "DTLZ2"
    # M = 2
    # Population, Boundary, Coding = P_objective.P_objective("init", Problem, M, particals)
    # max_ = Boundary[0]
    # min_ = Boundary[1]
    boundary = p_objective.get_hlf_boundary()
    max_ = boundary['Upper'].values
    min_ = boundary['Lower'].values

    mopso_ = Mopso(particals, max_, min_, thresh, mesh_div)  # 粒子群实例化
    pareto_in, pareto_fitness = mopso_.done(cycle_)  # 经过cycle_轮迭代后，pareto边界粒子
    np.savetxt("./img_txt/pareto_in.txt", pareto_in)  # 保存pareto边界粒子的坐标
    np.savetxt("./img_txt/pareto_fitness.txt", pareto_fitness)  # 打印pareto边界粒子的适应值
    print("\n", "pareto边界的坐标保存于：/img_txt/pareto_in.txt")
    print("pareto边界的适应值保存于：/img_txt/pareto_fitness.txt")
    print("\n,迭代结束,over")
    data = {}
    for name, row in zip(boundary['Name'], np.round(pareto_in)[:, -1]):
        data[name] = row.tolist()
    # with open('./output.yaml', 'w') as yaml_file:
    #     yaml.dump(data, yaml_file, default_flow_style=False)


if __name__ == "__main__":
    main()
