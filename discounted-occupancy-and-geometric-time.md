# 从式 (15) 到折扣占用分布：几何随机时刻的完整解释

> 整理日期：2026-08-15  
> 主题：式 (15)–(18) 的推导、几何分布的深层含义，以及它与旧 `beta` 风险方法的区别。

本文完整整理了这一段对话中的两个问题：

1. 怎样从逐时刻的 mean policy gradient，即式 (15)，引入式 (17) 的 normalized discounted state–action occupancy，并得到式 (18)？
2. 为什么归一化后的时间分布是几何分布？除了“它会产生 `eta^t`”之外，还有什么更深刻的概率、动态规划和 Markov 链解释？

---

## 一、原始问题：怎样从式 (15) 到式 (17)？

经过 likelihood-ratio policy gradient 和 tower property，mean-return 梯度被写成

\[
\nabla_\theta J(\theta)
=
\sum_{t=0}^{\infty}
\mathbb E_\theta
\left[
\eta^t G_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
\right].
\tag{13}
\]

对每一项取关于 \((S_t,A_t)\) 的条件期望：

\[
\nabla_\theta J(\theta)
=
\sum_{t=0}^{\infty}
\mathbb E_\theta
\left[
\mathbb E_\theta
\left[
\eta^t G_t
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
\mid S_t,A_t
\right]
\right].
\tag{14}
\]

由于 \(\eta^t\) 是确定量，而且

\[
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
\]

关于 \(\sigma(S_t,A_t)\) 可测，可以将它们移出内层条件期望：

\[
\nabla_\theta J(\theta)
=
\sum_{t=0}^{\infty}
\mathbb E_\theta
\left[
\eta^t
\mathbb E_\theta[G_t\mid S_t,A_t]
\nabla_\theta\log\pi_\theta(A_t\mid S_t)
\right].
\tag{15}
\]

定义 local mean target：

\[
Q_m^\pi(s,a)
:=
\mathbb E_\theta[G_t\mid S_t=s,A_t=a].
\tag{16}
\]

需要说明的是：**式 (17) 不是由式 (15) 推出的一个新结论，而是为了把式 (15) 中的无限加权求和改写成单个概率期望，专门引入的定义。**

### 1. 把被积函数记成一个整体

定义

\[
f_\theta(s,a)
:=
Q_m^\pi(s,a)
\nabla_\theta\log\pi_\theta(a\mid s).
\]

那么式 (15) 变成

\[
\nabla_\theta J(\theta)
=
\sum_{t=0}^{\infty}
\eta^t
\mathbb E_\theta[f_\theta(S_t,A_t)].
\tag{19}
\]

### 2. 展开每个时刻的期望

对于离散状态–动作空间，

\[
\mathbb E_\theta[f_\theta(S_t,A_t)]
=
\sum_{s,a}
\Pr_\theta(S_t=s,A_t=a)
f_\theta(s,a).
\]

代回式 (19)，并交换求和顺序：

\[
\begin{aligned}
\nabla_\theta J(\theta)
&=
\sum_{t=0}^{\infty}
\eta^t
\sum_{s,a}
\Pr_\theta(S_t=s,A_t=a)
f_\theta(s,a)
\\
&=
\sum_{s,a}
\left[
\sum_{t=0}^{\infty}
\eta^t
\Pr_\theta(S_t=s,A_t=a)
\right]
f_\theta(s,a).
\tag{20}
\end{aligned}
\]

连续状态或动作空间下，只需把求和换成相应的积分。

### 3. 为什么需要乘 \(1-\eta\)？

式 (20) 中括号里的量是所有时刻访问 \((s,a)\) 的折扣加权总量，但它还不是概率分布。其总质量为

\[
\begin{aligned}
&\sum_{s,a}
\sum_{t=0}^{\infty}
\eta^t
\Pr(S_t=s,A_t=a)
\\
&=
\sum_{t=0}^{\infty}
\eta^t
\sum_{s,a}\Pr(S_t=s,A_t=a)
\\
&=
\sum_{t=0}^{\infty}\eta^t
=
\frac{1}{1-\eta},
\end{aligned}
\]

其中要求 \(0\leq\eta<1\)。

因此乘上 \(1-\eta\) 将其归一化：

\[
\boxed{
d_\eta^\pi(s,a)
:=
(1-\eta)
\sum_{t=0}^{\infty}
\eta^t
\Pr_\theta(S_t=s,A_t=a)
}.
\tag{17}
\]

验证：

\[
\begin{aligned}
\sum_{s,a}d_\eta^\pi(s,a)
&=
(1-\eta)
\sum_{t=0}^{\infty}
\eta^t
\sum_{s,a}\Pr(S_t=s,A_t=a)
\\
&=
(1-\eta)
\sum_{t=0}^{\infty}\eta^t
=1.
\end{aligned}
\]

所以 \(d_\eta^\pi\) 是合法的 normalized discounted state–action occupancy distribution。

### 4. 代回去得到式 (18)

由式 (17)，

\[
\sum_{t=0}^{\infty}
\eta^t
\Pr(S_t=s,A_t=a)
=
\frac{d_\eta^\pi(s,a)}{1-\eta}.
\]

代入式 (20)：

\[
\begin{aligned}
\nabla_\theta J(\theta)
&=
\frac{1}{1-\eta}
\sum_{s,a}
d_\eta^\pi(s,a)
f_\theta(s,a)
\\
&=
\frac{1}{1-\eta}
\mathbb E_{(s,a)\sim d_\eta^\pi}
[f_\theta(s,a)].
\end{aligned}
\]

展开 \(f_\theta\)，得到

\[
\boxed{
\nabla_\theta J(\theta)
=
\frac{1}{1-\eta}
\mathbb E_{(s,a)\sim d_\eta^\pi}
\left[
Q_m^\pi(s,a)
\nabla_\theta\log\pi_\theta(a\mid s)
\right]
}.
\tag{18}
\]

因此，式 (15) 到式 (18) 的本质是

\[
\boxed{
\text{带 }\eta^t\text{ 的无限时间求和}
\Longleftrightarrow
\frac{1}{1-\eta}
\times
\text{归一化 occupancy 下的单个期望}
}.
\]

---

## 二、为什么归一化后恰好是几何分布？

式 (15) 原本已经包含 \(\eta^t\)。将这些时间权重归一化：

\[
p_t
=
\frac{\eta^t}{\sum_{k=0}^{\infty}\eta^k}
=
(1-\eta)\eta^t.
\]

这正是以 \(0\) 为起点的几何分布：

\[
\Pr(T=t)
=
(1-\eta)\eta^t,
\qquad t=0,1,2,\ldots
\]

所以正确的逻辑顺序是

\[
\boxed{
\text{原始目标已经给出 }\eta^t
\Longrightarrow
\text{归一化后得到几何分布}
}
\]

而不是先选择几何分布，再人为给目标加入 \(\eta^t\)。

若随机时刻 \(T\) 独立于环境轨迹，则

\[
\begin{aligned}
\Pr(S_T=s,A_T=a)
&=
\sum_{t=0}^{\infty}
\Pr(T=t)
\Pr(S_t=s,A_t=a)
\\
&=
(1-\eta)
\sum_{t=0}^{\infty}
\eta^t
\Pr(S_t=s,A_t=a)
\\
&=
d_\eta^\pi(s,a).
\end{aligned}
\]

因此 \(d_\eta^\pi\) 可以理解为：

> 独立抽取一个几何随机时刻 \(T\)，再观察轨迹在该时刻的 \((S_T,A_T)\)，所得的状态–动作分布。

---

## 三、更深刻的解释之一：随机终止

考虑一个随机生命周期模型：

- 每走一步，以概率 \(\eta\) 继续；
- 以概率 \(1-\eta\) 永久终止；
- 终止机制独立于环境轨迹。

若 \(T\) 是最后一个仍存活的时刻，则

\[
\Pr(T\geq t)=\eta^t.
\]

因此随机生命周期内的 undiscounted return 为

\[
\begin{aligned}
\mathbb E
\left[
\sum_{t=0}^{T}R_t
\right]
&=
\sum_{t=0}^{\infty}
\mathbb E[R_t\mathbf1\{T\geq t\}]
\\
&=
\sum_{t=0}^{\infty}
\Pr(T\geq t)\mathbb E[R_t]
\\
&=
\sum_{t=0}^{\infty}
\eta^t\mathbb E[R_t].
\end{aligned}
\]

所以 discounted return 可以等价理解为：

> 不对奖励进行时间折扣，但智能体每一步之后都以固定概率永久终止。

对于上述几何分布，

\[
\mathbb E[T]
=
\frac{\eta}{1-\eta},
\qquad
\mathbb E[T+1]
=
\frac{1}{1-\eta}.
\]

因此 \(1/(1-\eta)\) 也常被称为 effective horizon。比如 \(\eta=0.99\) 时，每一步有 \(1\%\) 的独立终止概率，平均访问步数为 \(100\)。

这是一种与折扣目标等价的数学解释，并不要求真实环境真的具有这种独立终止机制。

---

## 四、更深刻的解释之二：无记忆性与 Bellman 递推

几何分布最特殊的性质是 memorylessness：

\[
\Pr(T\geq t+1\mid T\geq t)
=
\eta.
\]

无论已经运行多久，下一步继续的概率始终相同。对应的终止 hazard 为

\[
\Pr(T=t\mid T\geq t)
=
1-\eta.
\]

这正好对应标准 discounted MDP 的时间齐次 Bellman 递推：

\[
G_t
=
R_t+\eta G_{t+1}.
\]

无论当前是第几步，未来子问题始终具有同样结构：

\[
\text{当前奖励}
+
\eta\times\text{同样形式的未来问题}.
\]

若生存函数满足乘法性质

\[
S(t+k)=S(t)S(k),
\]

那么离散时间下的非退化解就是

\[
S(t)=\eta^t,
\]

相应终止时间就是几何分布。因此

\[
\boxed{
\text{常数折扣}
\Longleftrightarrow
\text{常数终止概率}
\Longleftrightarrow
\text{几何生命周期}
\Longleftrightarrow
\text{时间齐次 Bellman 方程}
}.
\]

若采用非几何时间分布，下一步继续的概率通常依赖已经运行的时间，Bellman 方程会显式依赖 \(t\)。若想重新获得 Markov 性，一般需要把 clock 加入状态。

---

## 五、更深刻的解释之三：随机重启 Markov 链

令原策略下的状态转移矩阵为 \(P_\pi\)，初始状态分布为 \(\mu_0\)。折扣状态访问分布为

\[
d_{\eta,S}^\pi
=
(1-\eta)
\sum_{t=0}^{\infty}
\eta^t\mu_0P_\pi^t.
\]

利用矩阵几何级数：

\[
d_{\eta,S}^\pi
=
(1-\eta)
\mu_0(I-\eta P_\pi)^{-1}.
\]

它满足固定点方程

\[
\boxed{
d_{\eta,S}^\pi
=
(1-\eta)\mu_0
+
\eta d_{\eta,S}^\pi P_\pi
}.
\tag{21}
\]

定义重启转移核

\[
\widetilde P_\pi(s'\mid s)
=
\eta P_\pi(s'\mid s)
+
(1-\eta)\mu_0(s').
\]

这个新过程表示：

- 以概率 \(\eta\) 按原 MDP 转移；
- 以概率 \(1-\eta\) 重启到初始分布。

式 (21) 表明 \(d_{\eta,S}^\pi\) 是这个随机重启链的 stationary distribution。相应的状态–动作分布为

\[
d_\eta^\pi(s,a)
=
d_{\eta,S}^\pi(s)\pi_\theta(a\mid s).
\]

因此更准确地说：

- \(d_\eta^\pi\) 一般不是原始 Markov 链的 stationary distribution；
- 它是加入概率 \(1-\eta\) 重启后的链的 stationary distribution；
- 对原过程而言，它应称为 normalized discounted occupancy distribution。

---

## 六、Resolvent / Green function 解释

算子

\[
(I-\eta P_\pi)^{-1}
=
\sum_{t=0}^{\infty}
\eta^tP_\pi^t
\]

称为 Markov transition operator 的 resolvent，也可以理解为 discounted Green function。

其中

\[
\sum_{t=0}^{\infty}
\eta^tP_\pi^t(s,s')
\]

表示从 \(s\) 出发，在随机终止之前，对 \(s'\) 的折扣期望访问次数。

因此 discounted occupancy 的本质是

\[
\boxed{
\text{随机生命周期内的期望访问次数}
}.
\]

再乘 \(1-\eta\)，就是把期望访问次数归一化成概率分布。

---

## 七、几何分布不是唯一可能的随机时间分布

任何概率质量函数

\[
w_t\geq0,
\qquad
\sum_{t=0}^{\infty}w_t=1
\]

都可以定义

\[
d_w^\pi(s,a)
=
\sum_{t=0}^{\infty}
w_t\Pr(S_t=s,A_t=a).
\]

若 \(T\sim w\) 且独立于轨迹，则

\[
d_w^\pi(s,a)
=
\Pr(S_T=s,A_T=a).
\]

更一般地，如果原目标中的时间权重为 \(c_t\geq0\)，且

\[
C=\sum_{t=0}^{\infty}c_t<\infty,
\]

令

\[
w_t=\frac{c_t}{C},
\]

便有

\[
\sum_{t=0}^{\infty}
c_t\mathbb E[f(S_t,A_t)]
=
C\,
\mathbb E_{T\sim w}[f(S_T,A_T)].
\]

所以任何非负且可求和的时间权重都可以转化为随机时间分布。几何分布的特殊性不在于“只有它能定义 occupancy”，而在于：

1. constant hazard；
2. memoryless；
3. 无需把时间加入状态；
4. 产生时间齐次 Bellman 方程；
5. 对应固定概率重启的 Markov 链。

此外，在只指定非负整数随机变量的均值、而不引入其他时间结构时，几何分布还是最大熵分布。

---

## 八、与旧 \(\beta\) 风险方法的本质区别

### 1. Mean-return 中的 \(\eta^t\) 来自原始目标

mean-return 本身定义为

\[
J(\theta)
=
\mathbb E_\theta
\left[
\sum_{t=0}^{\infty}\eta^tR_t
\right].
\]

因此 mean gradient 中的 \(\eta^t\) 是原生权重。将其归一化为

\[
\Pr(T=t)=(1-\eta)\eta^t
\]

只是重新表达同一个目标，没有改变优化方向。

### 2. Chance-risk 梯度原本没有 \(\beta^t\)

正确的固定阈值风险梯度具有形式

\[
\nabla_\theta F_\infty(\theta;q)
=
\sum_{t=0}^{\infty}g_t,
\]

其中

\[
g_t
=
\mathbb E_\theta
\left[
\nabla_\theta\log\pi_\theta(A_t\mid\mathcal H_t)
A_t^c
\right].
\]

它原本没有 \(\beta^t\)。同时，非负整数集合上不存在可归一化的“均匀随机时刻”，因为

\[
\sum_{t=0}^{\infty}1=\infty.
\]

如果选择几何 proposal

\[
p_t=(1-\beta)\beta^t,
\]

为了仍然估计原始无权重和，必须做 inverse-probability correction：

\[
\begin{aligned}
\sum_{t=0}^{\infty}g_t
&=
\sum_{t=0}^{\infty}
p_t\frac{g_t}{p_t}
\\
&=
\mathbb E_{T\sim p}
\left[
\frac{g_T}{p_T}
\right]
\\
&=
\frac{1}{1-\beta}
\mathbb E_{T\sim p}
\left[
\beta^{-T}g_T
\right].
\end{aligned}
\]

正确 importance weight 是

\[
\frac{1}{p_T}
=
\frac{\beta^{-T}}{1-\beta},
\]

而不是 \(\beta^T\)。

若直接使用

\[
\sum_{t=0}^{\infty}\beta^tg_t,
\]

得到的是另一个时间加权 surrogate，而不是原始 chance-constraint 梯度。

所以二者的本质区别是

\[
\boxed{
\begin{aligned}
\text{Mean:}\quad
&\eta^t\text{ 来自原始 discounted-return 定义；}
\\
\text{Risk:}\quad
&\beta^t\text{ 是额外加入的，会改变原始 chance gradient。}
\end{aligned}
}.
\]

### 3. 为什么不能给风险 occupancy 直接设置 \(\eta=1\)

式 (17) 的归一化依赖

\[
\sum_{t=0}^{\infty}\eta^t
=
\frac{1}{1-\eta}<\infty.
\]

当 \(\eta=1\) 时，

\[
\sum_{t=0}^{\infty}1=\infty,
\]

而归一化因子 \(1-\eta\) 变成 \(0\)，因此不存在标准的 normalized undiscounted occupancy distribution。

这正是不能为了把风险梯度套入标准 discounted policy-gradient 形式，而人为添加 \(\beta^t\) 的原因。正确的 infinite-horizon 风险理论需要利用 conditional risk advantage 的阈值局部化和相应的可和性条件，而不是修改原始时间权重。

---

## 九、最终结论

从式 (15) 到式 (17)–(18) 的完整逻辑为

\[
\begin{aligned}
&\sum_{t=0}^{\infty}
\eta^t\mathbb E[f(S_t,A_t)]
\\
&=
\sum_{s,a}
\left[
\sum_{t=0}^{\infty}
\eta^t\Pr(S_t=s,A_t=a)
\right]f(s,a)
\\
&=
\frac{1}{1-\eta}
\sum_{s,a}
d_\eta^\pi(s,a)f(s,a)
\\
&=
\frac{1}{1-\eta}
\mathbb E_{(s,a)\sim d_\eta^\pi}[f(s,a)].
\end{aligned}
\]

几何分布的出现同时具有代数和结构意义：

\[
\boxed{
\eta^t
\Longleftrightarrow
\text{几何生存概率}
\Longleftrightarrow
\text{常数 hazard}
\Longleftrightarrow
\text{无记忆性}
\Longleftrightarrow
\text{时间齐次 Bellman 递推}
\Longleftrightarrow
\text{固定概率重启链}
}.
\]

对 mean-return 而言，这只是对原始折扣目标的等价表达；对原本没有时间折扣的 chance-risk 梯度而言，直接加入 \(\beta^t\) 会改变目标，不能用几何随机时刻来证明其正确性。
