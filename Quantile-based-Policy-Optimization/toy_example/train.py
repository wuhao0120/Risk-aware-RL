import os, re
import argparse
import torch
import datetime
import numpy as np
import random
import multiprocessing

from agents import PPO, PG, QPO, QPPO, SPSA
from envs import ToyEnv

# 创建一个参数管理类
# 这个类的作用是创建argparse解析器，并定义所有可配置的超参数（根据算法名称）
class Options(object):
    # 在实例化Options类时，需要传入算法名称algo_name，并自动调用__init__方法，获得该算法对应的参数配置
    def __init__(self, algo_name):
        # 创建一个argparse解析器对象，add_argument方法用于添加参数
        # 初始化时，会创建一个parser对象，传入默认参数，并根据算法名称添加额外参数
        parser = argparse.ArgumentParser()
        parser.add_argument('--env_name', type=str, default='ToyEnv')
        parser.add_argument('--algo_name', type=str, default=algo_name)
        parser.add_argument('--q_alpha', type=float, default=0.25)
        parser.add_argument('--est_interval', type=int, default=100)
        parser.add_argument('--log_interval', type=int, default=100)
        parser.add_argument('--max_episode', type=int, default=200000)
        parser.add_argument('--emb_dim', type=list, default=[8,8])
        parser.add_argument('--init_std', type=float, default=np.sqrt(1e-1))
        parser.add_argument('--gamma', type=float, default=0.99)

        # lr = a / (b + episode) ** c
        parser.add_argument('--theta_a', type=float, default=(10000**0.9)*1e-3)
        parser.add_argument('--theta_b', type=float, default=10000)
        parser.add_argument('--theta_c', type=float, default=0.9)
        parser.add_argument('--q_a', type=float, default=(10000**0.6)*1e-2)
        parser.add_argument('--q_b', type=float, default=10000)
        parser.add_argument('--q_c', type=float, default=0.6)
        
        # 创建一个Namespace对象，赋值给args变量，参数args=[]表示不从命令行读取参数
        # 如果不传参或者传入None，那么在命令行中启动脚本时，就要在命令后面指定参数
        # 第一哥parser.parse_args()用来检查算法类型，判断是否需要额外添加参数
        # 默认是args = self.parser.parse_args(args=[])，修改了方便从命令行直接读取参数
        args = parser.parse_args()
        if args.algo_name == 'QPPO':
            parser.add_argument('--lambda_gae_adv', type=float, default=0.95)
            parser.add_argument('--clip_eps', type=float, default=0.2)
            parser.add_argument('--vf_coef', type=float, default=0.5)
            parser.add_argument('--ent_coef', type=float, default=0.00)
            parser.add_argument('--upd_interval', type=int, default=2000)
            parser.add_argument('--upd_step', type=int, default=5)
            parser.add_argument('--mini_batch', type=int, default=100)
            parser.add_argument('--T', type=int, default=10)
            parser.add_argument('--T0', type=int, default=5)
        if args.algo_name == 'SPSA':
            parser.add_argument('--spsa_batch', type=int, default=5)
            parser.add_argument('--perturb_c', type=float, default=1.9)
            parser.add_argument('--perturb_gamma', type=float, default=1/6)
        if args.algo_name == 'PPO':
            parser.add_argument('--lambda_gae_adv', type=float, default=0.95)
            parser.add_argument('--clip_eps', type=float, default=0.2)
            parser.add_argument('--vf_coef', type=float, default=0.5)
            parser.add_argument('--ent_coef', type=float, default=0.00)
            parser.add_argument('--upd_interval', type=int, default=2000)
            parser.add_argument('--upd_step', type=int, default=10)
            parser.add_argument('--mini_batch', type=int, default=100)
        # 将配置好的parser对象赋值给类的parser属性
        self.parser = parser
     
    # Options类还有一个parse方法，用来在配置好的parser基础上，添加一些额外参数，如随机种子，device，日志路径，时间等
    # parse方法接收seed和device两个参数，返回一个包含所有参数的Namespace对象
    # 前面在配置超参数设置，parse方法在基础上添加额外的配置，返回最终的Namespace对象
    # 在算法中通过args.参数名来访问这些参数
    def parse(self, seed=0, device='0'):
        # 默认是args = self.parser.parse_args(args=[])，修改了方便从命令行直接读取参数
        args = self.parser.parse_args()
        args.seed = seed
        args.device = torch.device("cuda:" + device if torch.cuda.is_available() else "cpu")

        current_time = re.sub(r'\D', '', str(datetime.datetime.now())[4:-7])
        args.path = './logs/' + args.env_name + '/' + args.algo_name + '_' + current_time + '_' + str(args.seed)
        os.makedirs(args.path)
        return args

# run()是每个子进程执行的主入口函数
def run(algo_name, seed, device):
    # 创建Options对象，传入algo_name参数，这会在init中决定添加哪些参数
    # 调用parse方法，获得最终的配置
    # 返回一个完整的args对象，包含完整的超参数配置和运行时的信息
    args = Options(algo_name).parse(seed, str(device))

    # 让实验可以复现，固定三个随机源的随机种子
    # 把所有可能的随机元的种子都固定位传入的seed
    random.seed(args.seed) # python内置随机库的种子
    np.random.seed(args.seed) # Numpy的随机种子
    torch.manual_seed(args.seed) # Pytorch的随机种子，包括CPU张量操作

    env = ToyEnv(n=10) # 创建环境实例，参数n=10指定环境的维度

    if args.algo_name == 'PPO':
        agent = PPO(args, env)
    elif args.algo_name == 'QPPO':
        agent = QPPO(args, env)
    elif args.algo_name == 'PG':
        agent = PG(args, env)
    elif args.algo_name == 'QPO':
        """
        agent = QPO(args, env),在创建QPO类实例时,在init中会:
        - 创建Actor网络
        - 创建Memory回放缓冲区
        - 创建Adam优化器和学习率调度器
        - 创建Tensorboard SummaryWriter用于日志记录,选择args.path作为日志目录
        - 执行warm_up方法进行预热
        """
        agent = QPO(args, env)
    else:
        agent = SPSA(args, env)
    print(args.algo_name + ' running')
    # 调用agent的train方法，开始训练
    agent.train()


# 只有直接运行这个脚本时才会运行下面的代码
if __name__ == '__main__':
    # 并行进程数量
    n = 10
    # 创建一个长度为10的算法列表，所有元素都是SPSA
    # 意味着只跑SPSA算法的10个不同seed
    # 如果要跑多个算法的混合，改成algos = ['PPO','QPPO','PG','QPO','SPSA','PPO','QPPO','PG','QPO','SPSA']
    algos = ['SPSA']*n
    # 创建随机种子列表，从0到9
    seeds = [i for i in range(n)]
    # 创建设备列表，所有元素都是0，表示使用GPU 0
    # 如果有多个GPU，可以改为不同的设备号来指定并行的哪一个进程在哪个GPU上跑
    devices = [0 for i in range(n)]
    # 得到三元组列表，每个三元组包含(算法名称, 随机种子, 设备号)
    zipped_list = list(zip(algos, seeds, devices))
    # 创建一个进程池，包含n=10个worker进程，每一个worker的作用是执行run()函数
    pool = multiprocessing.Pool(processes=n)
    # starmap方法会将zipped_list中的每个三元组作为参数，传递给run函数
    # 上面创建的10个进程会在10个worker中并行执行
    pool.starmap(run, zipped_list)
    pool.close() # 关闭进程池，表示不再接受新的任务
    pool.join() # 等待所有worker进程执行完毕再继续



