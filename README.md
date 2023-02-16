### Hi there 👋
![My GitHub stats](https://github-readme-stats.vercel.app/api?username=CalvinDrinks&theme=cobalt)

![new][./new.md]

# Causal Reinforcement Learning

## 1 Introduction

人类从小天生能理解成因。认识到改变一个事件会导致另一个事情的发生。

因此我们可以积极干预身边的环境来达成目标或者获取新知识。

>Understanding cause and effect empowers us to explain behavior (Schult & Wellman, 1997), predict the future (Shultz, 1982), and even conduct counterfactual reasoning to eflect on past events (Harris et al., 1996). 
>
>These abilities are essential to the development of human intelligence, laying the foundation for modern society and civilization as well as advancing science and technology.



举例：人类对抗坏血病。![The_journey_to_discover_the_cause_of_scurvy](./The_journey_to_discover_the_cause_of_scurvy.png)

>It was only when the causality of scurvy was fully understood that effective solutions to combat the disease were discovered. 

这个例子说明了理解决策中的因果关系的重要性，以及忽视因果关系可能带来的灾难性后果。



只有数据是不能回答因果问题的。理解因果涉及提出有关数据产生的猜想和验证过程。基于数据的机器学习可以处理坏血病和柑橘类水果的关系，但不能实现因果推断。

> For exa轨道mple, if we replace citrus fruits with animal livers (also rich in vitamin C) in the scurvy prediction problem, the algorithm will probably give an incorrect prediction due to the significant differences in appearance and taste.

因果机器学习就是为了解决这个问题。

> In recent years, the combination of causality with machine learning has gained much attention and has been applied in various fields, including computer vision (Lopez-Paz et al., 2017; Shen et al., 2018; Tang et al., 2020; Wang et al., 2020b), natural language processing (Wu et al., 2021; Jin et al., 2021; Feder et al., 2022), and recommender systems (Zheng et al., 2021; Zhang et al., 2021b; Gao et al., 2022).

这些结果表明因果模型，显著提高了学习系统的分布鲁棒性和知识可迁移性。



不同于其他学习模式，强化学习积极干预环境，主动收集数据。所以强化学习自然连接了因果性。**但是大多数强化学习只能干预动作，导致其难以全面理解因果关系。**这一困难在离轨策略政策和离线环境中进一步加剧。



在强化学习中智能体不断的根据轨迹和错误优化策略。在这个动态过程中环境通过状态转移来回应动作，并返回奖励。状态转移和奖励分配都是因果关系。

>for example, vitamin C deficiency (the current state) causes scurvy (the next state), but not vice versa. Other environmental factors, such as the flavor and appearance of food, do not affect this transition. 

为了避免被非因果相关性所困扰，**智能体必须捕获驱动底层数据生成过程的因果关系**。否则，它将无法有效地学习，甚至陷入次优策略。



很多研究者研究了将因果知识集成到强化学习的原则方法。最受欢迎的是因果图，这是因果知识的一种定性形式。

>Causal graphs can be used to represent high-level, coarse-grained data generation processes that do not distinguish the meaning of each dimension, such as a standard Markov decision process (MDP). Meanwhile, causal graphs can also convey low-level, fine-grained causal knowledge, such as decomposing states into multiple variables
>according to their causal relations.

因果知识还可以被结构因果模型框架（SCM）定量表示，将在第二节展开介绍。结构因果模型SCM将数据生成的过程，看作为**有序的方程式集合以结构化的方式生成数据**。

正如我们稍后在第4节中所演示的，配备SCM的RL智能体可以直接在**不与实际环境交互的情况下生成数据，实现反事实数据增强和策略评估**。



大量研究工作在不同的条件下展开了因果强化研究，如bandits和MDPs；在线强化学习和离线强化学习；有模型和无模型强化学习设置。这些研究不断展现了因果强化学习有表现的更好更稳定的倾向。但是这些研究没有很好的相互关联。基于因果在人类智能的核心角色，因果强化学习有潜力克服现有强化学习的问题和更困难的决策问题。

这篇文章旨在给出一个全面的因果强化学习综述，连接基于SCM框架的现存的工作，这将允许系统性和原则性的因果知识被整合到学习过程中。

文章的主要贡献如下：

> • Our survey of causal RL presents a comprehensive overview of the field, aligning existing research within the SCM framework. In particular, we introduce causal RL by answering three fundamental questions: What it is causal RL? Why does it need to be studied? And how does causal modeling improve existing RL approaches? We also present a clear and concise overview of the foundational concepts of causality research and RL. To the best of our knowledge, this is the first comprehensive survey of causal RL in the extant literature on RL 1 .
>
> • We identify the bottleneck problems in RL that can be solved or improved by means of causal modeling. **We further propose a problem-oriented taxonomy. This taxonomy will help RL researchers gain a deeper understanding of the advantages of causal modeling and the opportunities for further research.** On the other hand, RL practitioners can also benefit from this survey by identifying solutions to the challenges they face. Additionally, we compare and analyze existing causal reinforcement learning research based on their techniques and settings.
>
> • We highlight major unresolved issues and promising research directions in causal RL, such as **theoretical advances, benchmarks, and specific learning paradigms**. These research topics will become increasingly important in the coming years and will help advance the use of RL in real-world ap-plications. Therefore, having a common ground for discussing these valuable ideas in this emerging field is crucial and will facilitate its continued development and success.



## 2 Background

为了更好地理解因果RL，这是一个结合了因果研究和强化学习优势的新兴领域，我们引入了一些基础和两个研究领域共同概念。

### 2.1 A Brief Introduction to Causality

数学语言上，有SCM和PO(potential outcome) 两种框架。我们专注于前者因为其提供了一种图形方法，可以帮助研究者总结和理解数据生成的过程。值得注意的是从逻辑上两种框架是等价的，且很多假设可以相互交换。

**Definition 2.1** (Structual Causal Model). 一个SCM可以表到为一个四元组 $(V, U, F, P (\mathbf U))$, 其中： 
• $\mathcal V = {V_1 , V_2 , · · · , V_n }$ 是研究问题中感兴趣的一组**内生变量**（endogenous variables）。
• $\mathcal U = {U_1 , U_2 , · · · , U_n }$ 是一组**外生变量**（exogenous variables） 相对于$\mathcal V$, 他通常是不可见的，代表随机来源，比如误差。
• $\mathcal F = {f_1 , f_2 , · · · , f_n }$ 是一组结构方程，用于给$\mathcal V$中的每一个变量赋值。
• $P (\mathbf U)$ 是外生变量在$\mathcal U$的联合概率分布 .

![SCM_and_the_causal_graph](./SCM_and_the_causal_graph.png)

**结构因果模型**

在definition 2.1中，外生变量$\mathbf U$是整个系统的输入和不确定来源。换句话说**所有外生变量确定后，内生变量也就被响应决定了**。

任何结构方程$f_i ∈\mathcal F$ 给一个内生变量赋值$V_i$ ，其自变量由相关外生变量和其他内生变量 ($V_i$的成因）构成。通过这种方式，结构方程在数学上描述了变量之间的因果关系和数据生成过程背后的因果机制。通过去顶外生变量的分布$P(\mathbf U)$和顺序执行所有的$\mathcal F$里的结构方程，任何样本都可以从联合分布$P(\mathbf V)$中生成。



**因果图**

每个SCM都可以与因果图$\mathcal G = {\mathcal V, \mathcal E}$关联，其中节点$\mathcal V$表示内生变量，边$\mathcal E$表示由结构方程确定的因果关系。一般来说，因果图是有向无环图。从节点$V_j$到节点$V_i$存在一条边$e_{ij}∈\mathcal E$。

$V_j$是变量$V_i$的结构方程$f_{V_i}$中的自变量。换句话说，如果$V_j$是SCM中$V_i$的结构方程的一部分，则认为它是因果图上$V_i$的父节点，即:$V_j∈\mathbf P\mathbf A (V_i)$。图2a显示了坏血病问题的SCM(简化版)和相应的因果图。在复杂的问题中，结构方程往往是未知的，而我们通常有因果图的先验知识，这对于解决许多因果推理问题已经足够了。图2b显示了因果图的三个基本构件，即链（Chain）、叉（Fork）和碰撞（Collider）。这三个简单的结构可以组合起来创建更复杂的数据生成过程。



**干预**

干预不是被动地观察数据，而是积极参与数据生成过程的方法。本质上，干预创造了新的数据分布，或者换句话说，所有分布变化都是数据生成过程中不同干预的结果。有两种类型的干预：一种是硬干预，直接将变量设为固定常值；另一种是软干预，在对结构方程进行更改的同时保留一些原始依赖关系(例如，改变外生变量的分布)。

> For example, finding a way to prevent scurvy is essentially about identifying effective interventions (through food or medicine) that lower the probability of getting scurvy. 

$P(Y \mid do(X) = x)$表示干预概率，表示变量$X$固定为$x$时，结果变量$Y$的概率分布。图3a说明了条件概率和干预概率的差异。



**反事实**

反事实思考是关于“如果”的问题，例如”假如坏血病患者如果吃了足量柑橘类水果会怎么样？他们是否会保持健康？“这种思考过程出现在我们的日常生活中，让我们反思自己在过去事件中的行为，并最终做出改进。

在研究社区中，反事实变量通常用下标表示，例如$Y_{X=x}$(或没有歧义时为$Y_X$)。这种写法有助于研究人员将反事实变量与原始变量$Y$区分开来。基于这种形式，反事实推理旨在估计概率，如$P(Y_{X=1}\mid X = 0, Y =1)$。我们可以将反事实推理视为创建一个不同于事实世界的虚构世界，而干预只研究事实世界。特别地，如果将相同干预施加在事实和想象世界，它们会重叠$P(Y_{X=0}\mid X =0, Y = 1) =P(Y\mid do(X = 0))$。图3b为反事实推理的可视化表示。

![condition_intervention_counterfact](./condition_intervention_counterfact.png)

**因果发现和因果推理**

在因果关系的研究中，有两个主要的重点领域：因果发现和因果推理。因果关系发现涉及使用感兴趣的变量的数据来推断它们之间的因果关系（换句话说，确定这些变量的因果图）。传统的方法使用条件独立性验证来推断因果关系，最近一些研究基于大型数据集使用深度学习技术。Glymour等人(2019)和Vowels等人(2022)全面综述了因果发现领域。

另一方面，因果推理研究如何在给定的因果模型下估计因果效应，如干预概率。干预包括积极地操作系统或环境，这可能昂贵且有潜在危险（例如，在医学实验中测试一种新药）。因此，因果推理的一个核心挑战是如何将因果效应转化为可以从观察到的数据中推断出来的统计估计。给定因果图后，通过使用do-calculus系统地确定因果效应的可识别性(Pearl, 1995)。

**因果因式分解**

因果图通过一个联合概率分布的因果方向，提供了一个因式分解方法，这称为因果因式分解：

$$
P(\mathbf V) = \prod^{n}_{i=1}P(V_i∈\mathbf P\mathbf A (V_i))\tag{1}
$$

其中（条件）概率分布$P(V_i\mid \mathbf P\mathbf A(V_i))$被称为因果机制。**因果因式分解是数据生成过程所特有的**，但其他形式的因式分解也可以满足联合分布所体现的统计依赖关系。以图2为例。数据生成过程采用链式结构。联合分布$P(X, Y, Z)$可以因式分解为$P(X)P(Z\mid X)P(Y \mid Z)$或$P(X)P(Y\mid X)P(Z\mid X, Y)$。前者是因果关系，后者不是。由于因果图结构的稀疏性，因果因式分解允许更有效的学习和推理。

此外，因果因式分解有助于归纳新问题。考虑对维生素C摄入的干预，这会改变联合分布。具有因果因式分解的智能体只需要调整一个模块$(P(Z\mid X))$来预测新分布下的坏血病，而非因果模型则必须重新学习两个模块$(P(Y\mid X)$和$P(Z\mid X, Y))$。这是因为干预只影响维生素C摄入的因果机制，与患坏血病的概率无关。这个性质被称为模块化(Pearl, 2009b)，或独立因果机制(Schölkopf et al., 2021)，表明因果生成过程由一系列稳定且自主的模块(因果机制)组成。更改其中一个模块并不会影响其他模块。此外，分布的微小变化通常只涉及因果分解中的几个模块，为设计高效的机器学习算法和模型提供了原则。

### 2.3 Causal Reinforcement Learning

在强化学习中MDP可以代表一段数据生成过程。从因果推断的角度我们可以将MDP转换成一段SCM。条件转移方程和奖励可以被描述为带有外生变量的确定函数，通过SCM的结构方程$\mathcal F$描述。初始状态可以被理解为包括$\mu_0$外生变量$P_{\mathcal U}$。这种转换适用于任何MDP，可以通过自回归统一化（也称为重新参数化(reparameterization)技巧来实现(Buesing et al., 2019)。

因果图和SCM与MDP的对应性在图4中展示。值得注意的是策略$\pi$不是一个因果关系；这是一个软干预，它保留了对状态变量的依赖。即$\operatorname{do}(a \sim \pi(\cdot \mid s))$。如我们前面所讲到，干预是生成了不同的数据分布；因此不同的策略导致了不同的轨迹分布和不同的预期回报。很明显，同轨策略从干预数据中学习，使其能够直接学习到受动作影响的因果效应估计值。与之对比的是，离轨策略和离线强化学习涉及智能体被动观测和学习过往策略收集到数据。在这种情况下数据的采集是收学习者观测所影响的，容易受到虚假关联的影响。

![causal_MDP](./causal_MDP.png)

SCM框架允许我们讨论决策问题的因果性，理解不同分布是怎样产生的，并且将因果知识以清晰和可重用的方法组织起来。更进一步，SCM允许我们探索RL中的反事实问题，这在非因果方法下是不能实现的。这篇文章我们将以下面的方式定义因果强化学习。



**Definition 2.3** (Causal reinforcement learning)

因果RL是一种RL方法的涵盖性术语，它专注于理解和利用底层数据生成过程的因果机制来为决策提供信息。

定义2.3体现了因果强化学习不同于其他形式强化学习，其注重因果关系而不是训练数据的表面相关关系或模式。我们注意到，在RL中，智能体试图找到具有最高预期回报的策略，而**不仅仅是推断干预的因果效应**。在下一节中，我们将证明因果模型不仅可以提高对因果关系的理解，而且在非因果方法效果不好的问题上有效。



## 3 Why Causality Is Important in Reinforcement Learning

强化学习在过去十年间获得飞速发展，但是仍旧面临挑战。我们会总结四个主要的将强化学习大规模应用于现实世界的阻碍，并且因果模型都可以给出有潜质的解决方案。我们分析了这些挑战并且解释为什么它们可以从因果模型中获益。

### 3.1 Sample Efficiency in Reinforcement Learning

#### 3.1.1 The Issue of Sample Efficiency in Reinforcement Learning

在RL中，用于训练的数据是事先不提供的。与直接从固有数据集中学习的有监督和无监督学习方法不同，RL智能体需要主动收集新数据以优化其策略以获得最高回报。一个有效的RL算法应该能够用尽可能少的经验掌握最优策略（换句话说，它必须是样本高效的）。目前的方法通常需要收集数百万个样本才能成功完成简单的任务，更不用说更复杂的环境和奖励机制了。

> For example, AlphaGo Zero was trained over roughly $3\times 10^7$ games of self-play (Silver et al., 2017); 
>
> OpenAI’s Rubik’s Cube robot took nearly 104 years of simulation experience (OpenAI et al., 2019). 

这种低效率带来了很高的训练成本，并阻止了使用RL技术进行求解现实世界里快速变化的决策问题。因此，样本效率问题是RL的核心挑战，开发能够节省时间和资源的样本高效RL算法至关重要。

#### 3.1.2 Why Causal Modeling Helps Improve Sample Efficient?

RL的研究人员一直关注样本效率(Kakade, 2003;Osband et al., 2013;Grande et al., 2014;Yu,2018)。这个问题通常与学习过程中的几个要素有关。

第一个是**抽象**，这也是机器学习中的一个基本问题。由于降低了维数，适当抽象的问题更容易解决。**在抽象空间中理解环境和学习策略会更高效。**一些方法(Jong & Stone, 2005;Zhang et al., 2022)通过聚合多个状态来实现抽象。虽然这些方法通过消除不相关的变量有效地降低了维数，但其余变量之间仍然存在冗余依赖关系，容易产生虚假相关性。因果模型通过识别环境动态中的因果变量并仅保留必要的依赖关系来简化问题的复杂性。因此，因果建模可以实现更好的抽象，帮助智能体专注于核心方向，间接提高样本效率。此外，抽象与表示学习密切相关(Schölkopf et al., 2021)。因果表示比基于相关性的表示具有更好的鲁棒性和可转移性此外，抽象与表示学习密切相关(Schölkopf et al., 2021)。此外，抽象与表示学习密切相关(Schölkopf et al，2021)。因果表示比基于相关性的表示具有更好的鲁棒性和可转移性。

另一个常见思路是**设计更有效的探索方法**，以帮助智能体收集最有利于策略学习的数据(Yang et al., 2022b)。一些研究(Pathak et al, 2017;Burda et al., 2022)利用了发展心理学中的内在动机概念(Ryan & Deci, 2000;Barto, 2013)通过为探索行为提供内在奖励来激励智能体探索未知环境。这种方法向智能体提供延迟反馈，不像收集数据时直接鼓励探索的方法。基于不确定性的方法遵循面对不确定性时的乐观原则(Ciosek et al., 2019;Lee et al., 2021a)。它鼓励智能体优先探索高认知不确定性的区域（将不确定性视为奖励），以较少的交互轮次定位较高奖励的区域。然而，并非所有高度不确定性的地区都同样重要。因果建模有助于智能体检测关键区域，例如通过缩小范围，确定机器人操作中靠近目标的区域。

除了简化问题和收集信息数据，样本效率还包括更有效地使用数据。基于模型的强化学习(MBRL) (Wang et al., 2019;Luo et al., 2022)是一个直接的想法。只要对环境模型的学习是准确和高效的，智能体就可以在不访问环境的情况下使用学习到的模型生成数据。因果模型比传统的基于相关性的模型更强大，因为它们更鲁棒，能够进行反事实推理。通过明确地考虑外生变量，因果模型可以产生更高质量的样本。

### 3.2 Generalizability in Reinforcement Learning

#### 3.2.1 The Issue of Generalizability in Reinforcement Learning

RL的可泛化性是现实世界中RL算法部署的另一个主要挑战。它指的是训练有素的策略在新的、看不见的情况下表现良好的能力(Kirk et al., 2022)。在同一个环境中进行训练和测试一直是RL社区中一个臭名昭著的问题(Irpan, 2018)。虽然人们通常期望RL在不同(但相似)的环境或任务中可靠地工作，但传统的RL算法通常设计用于解决单个MDP。它们很容易在环境中过拟合，无法适应微小的变化。

即使在相同的环境中，RL算法也可以使用不同的随机种子产生差异很大的结果(Zhang et al., 2018a;b)，表明不稳定性和过拟合。Lanctot等人(2017)提出了一个多智能体场景中过拟合的例子，其中训练有素的RL智能体在对手改变其策略时难以适应。Raghu等人(2018)也观察到了类似的现象。此外，现实世界是非静止的，不断变化的(Hamadanian et al., 2022)，所以一个好的RL算法必须是健壮的，以处理这些变化。当情况有所变化，智能体应该迁移他们的技能而不是从头再来。

![typesof_generation_problem](./typesof_generation_problem.png)

#### 3.2.2 Why Causal Modeling Helps Improve Generalization and Facilitate Knowledge Transfer?

在RL中，泛化涉及到不同的样本或不同的环境。Kirk等人(2022)提出使用上下文(contextual)的MDP (CMDP) (Hallak et al., 2015)来形式化RL中的泛化问题。CMDP类似于标准MDP，但是它捕获由上下文决定的一组环境或任务中的可变性。这些上下文变量可能会以不同的方式影响状态空间、转换函数、奖励函数或发射函数(将状态映射到观测值)。它们可以代表各种因素，如随机种子、目标、颜色和游戏关卡的难度。

之前的一些研究表明，数据增强可以改善泛化(Lee et al.， 2020;Wang et., 2020a;Yarats et al., 2021)，特别是基于视觉的控制。这个过程包括通过随机移动、混合或扰动观测来生成新数据，这使得学习策略更能抵抗类似的变化。另一种常见做法是域随机化。在模拟到真实的强化学习中，研究人员随机化模拟器的参数，以促进对现实的适应(Tobin et al., 2017; Peng et al., 2018)。受域随机化的启发，Wellmer & Kwok(2021)将dropout技术应用于世界模型，创造无限的梦想世界。OpenAI开发了一种自动的域随机化，成功解决了魔方问题(OpenAI et al., 2019)。此外，一些方法试图通过设计特殊的网络结构来整合归纳偏差，以提高泛化性能(Kansky et al., 2017; Higgins et al., 2017; Zambaldi et al., 2019; Raileanu & Fergus, 2021)。

尽管这些方法都显著的提升了泛化性，但是这种泛化性和底层数据生成的关系依旧不清楚，但这种关系对于识别变化因素至关重要。从因果关系的角度来看，分布内泛化不涉及操纵数据生成过程。只要SCM足够精确，它就能阻止智能体在非因果关联上的过拟合出现，并且天然的使得智能体对相同因果模型的生成的数据具有泛化能力。这告诉我们智能体可以无需训练就在没见过的数据上表现得很好。

在分布外(OOD)泛化场景中，数据分布发生了变化，这可以解释为数据生成过程中的外部干预。不同的干预措施导致不同的数据分布，这可能对泛化提出不同的挑战。图5举例说明了对应于不同干预措施的泛化问题。因果建模使我们能够明确地分析和讨论数据分布中的变化，区分哪些变化和哪些保持不变。当有更多的领域知识时，我们可以进一步分解内生变量，以获得更细粒度的数据生成过程（这也可以通过因果学习来实现）。因果建模对数据沿因果方向的联合分布进行了因子分析（见公式1）。一般来说，数据分布的变化只与少量的因果机制有关。根据独立因果机制原则(Schölkopf et al., 2021)，主体只需要调整一些模块以适应新的环境或任务。其他模块保持不变，可直接重用。特别是，当改变的模块与任务无关时(例如，机器人操作任务中的背景颜色)，我们可以训练一个只关注不变性的策略。在这种情况下，智能体可以在任何新的情况下直接重用策略，实现zero-shot泛化。

![graph2_correlations](./graph2_correlations.png)

### 3.3 Spurious Correlations in Reinforcement Learning

#### 3.3.1 The Issue of Spurious Correlation in Reinforcement Learning

仅从数据中学习决策是不够的，因为相关性并不意味着因果关系。伪相关是两个变量之间的关系，看起来是因果关系，但实际上是由第三个变量引起的，给学习问题带来了不必要的偏差。这种现象在机器学习应用中广泛存在，下面给出了几个典型的例子。

-  In recommendation systems, both user behavior and preferences are influenced by conformity. If the recommender ignores conformity, it may overestimate a user’s preference for certain items (Gao et al., 2022); 
-  In image classification, if dogs frequently appear with grass in the training set, the classifier may label an image of grass as a dog. This is because the model relies on the background (irrelevant factors) instead of the pixels corresponding to dogs (the actual cause) (Zhang et al., 2021a; Wang et al., 2021c);
-  When determining the ranking of tweets, the use of gender icons in tweets is usually not causally related to the number of likes; their statistical correlation comes from the topic, as it influences both the choice of icon and the audience. Therefore, it is not appropriate to determine the ranking by gender icons (Feder et al., 2022).

如果我们想在现实场景中应用RL，重要的是要注意伪相关性，特别是当智能体处理有偏差的数据时。

> For instance, when optimizing long-term user satisfaction in multiple-round recommendations, there is often a spurious correlation between exposure and clicks in adjacent timesteps. This is because they are both influenced by item popularity. From another perspective, when we observe a click, it may depend on user preference or item popularity, which creates a spurious correlation between the two factors. 
>
> In both scenarios, if the agent ignores causality, it will make incorrect predictions or decisions, such as by creating filter bubbles by only recommending popular items (a suboptimal policy for both the system and the user).

简而言之，如果智能体学习到两个变量之间的虚假相关性，它可能会错误地认为改变一个变量会影响另一个变量，即使在底层数据生成过程中它们之间没有因果关系。这种误解会导致次最优的甚至有害的行为。

#### 3.3.2 Why Causal Modeling Helps Address Spurious Correlations

从因果关系的角度来看，当数据生成过程涉及未观察到的混杂因素(常见原因)或当碰撞节点(常见结果)作为条件时，就会出现虚假相关性。前者导致混杂偏差，后者导致选择偏差。有关这些现象的可视化解释，请参见图6。使用因果图，我们可以通过仔细检查数据生成过程来追踪虚假相关性的来源。

为了消除偏差，有必要利用因果关系，而不是依赖统计相关性。这是因果推理强项：它提供了工具，使我们能够分析和处理混淆和选择性偏差(Pearl, 2009b; Glymour et al., 2016)，帮助RL智能体更准确地估计决策问题的因果影响。因此，因果建模有助于避免对环境或任务的错误解释，从而导致次优策略。使用因果图和因果推理使智能体在决策中提高推理的正确性。

### 3.4 Considerations Beyond Return

一般来说，RL关注的是收益最大化。然而，随着基于RL的自动化决策系统在我们的日常生活中变得越来越普遍，关注智能体如何与人互动以及它们如何影响社会是至关重要的。

#### 3.4.1 Explainability in Reinforcement Learning

强化学习的可解释性指的是强化学习的决策可以被理解和解释的能力。这对于研究者和普通用户来说都是重要的。可解释性反映了智能体学习到的知识，有助于深入理解，允许研究人员高效参与算法设计和持续优化。此外，解释提供了决策过程的内在逻辑。当智能体超过人类水平，我们可以从解释中提取知识，指导人类在特定领域的实践。对于普通用户，可解释性展现了决策的理由，因此让用户对于智能体有了跟深入的理解，这增强了使用者的信心。

#### 3.4.2 Achieving Explainability through Causal Modeling

可解释的RL方法可以分为两类：事后方法和内在方法(Puiutta & Veith, 2020; Heuillet et al., 2021)。前者在执行后提供解释，而后者本质上是透明的。事后解释通常是基于相关性建立的，例如显著性图方法(Greydanus et al., 2018; Mott et al., 2019)。正如我们前面提到的，基于相关性的结论可能是不可靠的，不能回答因果问题。另一方面，内在解释可以使用易于理解的算法来实现，如线性回归或决策树(Coppens et al., 2019)。然而，有限的模型能力可能不足以解释复杂的行为(Puiutta & Veith, 2020)。

人类拥有一种与生俱来的强大能力，可以通过一种方法在不同的事件之间建立联系“心理因果模型(Sloman, 2005)”。在日常生活中，我们经常使用因果语言，例如“因为”、“因此”和“如果……就好了”，以促进沟通和协作。使用因果模型可以进行自然而灵活的解释，因为它不需要选择特定的算法或模型。在RL中，基于因果关系的可解释性为智能体的决策提供了稳定的支持，并帮助我们理解智能体如何解释环境和任务。当智能体出错时，我们可以用更量身定制的解决方案来应对。

#### 3.4.3 Fairness in Reinforcement Learning

随着机器学习在我们的日常生活中变得越来越普遍，企业所有者、普通用户和政策制定者等利益相关者正在意识到公平的重要性。这一概念适用于任何类型的自动化系统和决策支持系统，包括基于RL的系统。特别是，RL智能体应该努力真正造福于人们，促进社会公益，而不是对特定的个人或群体造成歧视或伤害。此外，现实世界中的公平问题通常是动态的(Gajane et al., 2022)，涉及多个决策。

> For example, a hiring process is typically a sequential decision process, and the actions may have cumulative effects on fairness. Ignoring the dynamic nature of a system may lead to unintended unfairness (Liu et al., 2018; Creager et al., 2020; D’Amour et al., 2020).

#### 3.4.5 Safety in Reinforcement Learning

研究人员经常使用约束MDPs (Altman, 1995;1999)，以模拟RL中的安全问题。通过合并表示安全问题的约束集，限制型MDPs扩展了MDPs。因此，大多数相关研究都集中在解决约束优化问题上(Achiam et al., 2017; Chow et al., 2017)，很少考虑因果关系。借助因果模型，我们可以通过分析不安全状态或行为的生成过程，更有效地形式化先验知识，如解释，并获得有价值的见解。当RL智能体违反安全约束时，因果模型可以帮助研究人员和专家更好地理解意外结果的原因，并设计解决方案以防止它们再次发生。此外，因果模型可用于反事实策略评估，允许在实际应用程序中部署RL智能体之前测试和识别潜在的安全问题。

总的来说，因果模型有助于确保RL技术被安全、负责地使用，避免灾难性的后果。



**To summarize**: 我们在本节中讨论了RL的几个关键挑战，并考虑了为什么因果模型在解决或减轻这些挑战方面起着至关重要的作用。接下来，我们回顾了因果RL的最新进展。



## 4 Existing Work Relating to Causal Reinforcement Learning

在前一节中，我们强调了因果RL的重要性。然而，关于这一主题的现有文献缺乏清晰度和连贯性，主要是因为因果模型更多的是一种心态，而不是具体问题。它为解决广泛的问题提供了原则和见解。本节回顾了现有的因果RL方法，这些方法解决了第3节中概述的四个关键挑战。

> We organize these approaches based on their problem settings and solution methods with the goal of better understanding their connections and relationships

![table1](table1.png)

### 4.1 Causal Reinforcement Learning for Addressing Sample Inefficient

因果模型为设计有效抽样RL算法提供了一些有用的原则。我们可以将这些原则组织成三个研究方向：表示学习、定向探索和数据增强。代表作品见表1。

#### 4.1.1 Representation Learning for Sample Efficiency

一个好的表示对于RL样本效率是有益的。通过给出一个紧凑并富含信息的环境表示，RL智能体可以从更少的样本中高效地学习。这是因为好的表示可以帮助智能体分清楚环境中的重要特征并且抽象出不必要的细节，允许智能体学习到更泛化的策略，并更好地利用经验。

Sontakke等人(2021)的研究涉及对具有不同物理性质的各种环境生成的轨迹进行聚类，并使用聚类结果作为环境的因果表示。该学习策略利用由因果表示增强的状态，表现出出色的零次概化能力，并且只需要少量的训练样本就能在新环境中收敛。

因果模型还激发了状态抽象。Lee等人(2021b)**使用干预来识别对成功完成任务很重要的状态变量**，降低了状态空间的维数，简化了问题。状态抽象的另一种方法是**因果动力学学习**，它通过**学习因果动力学模型来识别不同变量之间的因果关系**。Huang等人(2022b)提出使用充足行动状态表示(包含足够决策信息的最小状态变量集)来增强RL的岩本效率。王等人(2022)则研究了任务无关的状态抽象。与以前的工作不同，他们采用了共享的结构化动态模型，删除了不相关的依赖关系，同时保留了可能在新任务中使用的相关状态变量。为了在下游任务中设计出规划或基于模型的RL，智能体只需要学习一个奖励预测器。

#### 4.1.2 Directed Exploration for Sample Efficiency

虽然良好的环境表示是有益的，但对于样本效率来说，这并不一定足够RL (Du et al., 2020)。为了提高样本效率，研究人员一直在研究定向探索，即引导智能体探索状态空间中被认为信息量更大或更有可能产生高回报的特定部分的策略。这可以通过奖励那些发现新奇或不确定状态的探索性行为来实现。从因果关系的角度来看，并非所有高度不确定性的地区都是同等重要。只有那些与任务成功形成因果关系的因素才值得探索。

![eg_counter_data](./eg_counter_data.png)

> As an example, Seitzer et al. (2021) studied the problem of directed exploration in robotic manipulation tasks. Valuable data can be generated only if the agent touches the target object, a prerequisite for learning complex manipulation tasks. 
>
> The authors proposed a method of measuring the effect of an action on the object (a causal quantity) and incorporating it into exploration, greatly improving the sample efficiency of robotic manipulation tasks. 

此外，智能体对因果关系的好奇心可以驱使它在没有外部奖励的情况下获得有用的行为和知识。受独立因果机制原则的启发，Sontakke等人(2021)提出了一种将因果好奇心形式化的方法，允许主体执行有助于理解环境的实验。实验表明，经过因果好奇心预训练的RL智能体可以更快地学习解决新任务。

#### 4.1.3 Data Augmentation for Sample Efficiency

数据增强是一种常见的机器学习技术，旨在通过生成额外的训练数据来提高算法的性能。反事实数据增强是一种基于因果关系的方法，它使用因果模型来模拟环境并生成在现实世界中未观察到的数据。这对于RL问题特别有用，因为收集大量的现实数据通常是困难或昂贵的。通过模拟不同的反事实场景，RL智能体可以在不与环境交互的情况下确定不同动作的效果，使得学习过程更具有样本效率。

反事实数据增强的实现遵循一个由三个步骤组成的反事实推理程序(Pearl, 2009a)，如图7所示：

1. **诱导**(Abduction)是利用观测数据推断外生变量$\mathcal U$的值。
2. **动作**包括改变SCM感兴趣变量的等式结构。
3. **预测**是利用改进的SCM通过将外生变量代回方程计算，生成反事实数据。

虽然MBRL方法也可以用学习到的模型生成样本，但它们缺乏对外生变量建模的方法。当外生变量分布复杂时，这可能导致欠拟合(Buesing et al., 2019)。相反，反事实数据增强使用SCM框架明确地考虑外生变量，并能够生成更高质量的样本。从贝叶斯的角度来看，传统的MBRL方法使用外生变量的固定先验分布，而反事实数据增强使用观测数据来估计后验分布。

### 4.2 Causal Reinforcement Learning for Addressing Generalizabiliy

现实世界中的决策问题是不断变化的，很难预测。因此，RL算法必须能够在部署过程中在新的和看不见的情况下表现良好，也称为泛化。

泛化涉及许多类型的问题。zero-shot泛化要求智能体只在训练环境中学习，并在未见的环境中进行测试。虽然这种设置很吸引人，但在实践中有时是不可行的。或者，适应(Zhang et al., 2015;Gong et al., 2016)假设智能体可以在测试域接受额外的培训，包括各种设置，如迁移RL (Zhu et al., 2020)，多任务强化学习(Vithayathil Varghese & Mahmoud, 2020)，或终身强化学习(Khetarpal et al., 2022)。大量的研究都考虑了RL泛化 (Kirk et al., 2022)，但仍然缺乏对智能体需要什么能力来实现泛化以及什么样的泛化可以从学习算法中被预期的理解。因果模型为回答这些问题提供了一种可能的方法。本节根据现有因果RL算法的具体泛化目标进行分类。代表作品见表2。

![table2](table2.png)

#### 4.2.1 Generalize to Different Environments

首先，我们考虑如何对不同环境的泛化。从因果关系的角度来看，不同环境的大部分因果机制相同，但在某些模块上有所不同，这是由于对状态变量的不同干预造成的。基于这些变量之间的因果关系，我们可以进一步将现有作品分为两类：对不相关变量的泛化和对不同动态的泛化。

为了提高对不相关因素的归纳能力，RL智能体**必须掌握数据生成过程中的不变性**。

- Zhang等人（2020a）研究了在块状MDP框架内泛化到不同观察空间的问题。这个问题在现实中很常见，比如当机器人配备了不同类型的摄像机和传感器。在块状MDP框架内，观察可能是无限的，但它可以唯一地确定环境的状态（有限但不可观察）并保持马尔科夫特性。作者提出使用不变预测来学习奖励的因果表征，这使得智能体能够在新的观察空间中实现zero-shot泛化。

- Bica等人（2021b）引入了不变因果模仿学习（ICIL），允许从多个领域学习模仿政策，然后在新环境中部署。ICIL方法通过学习一个跨领域一致的因果变量的共享表示来实现这一目标。

- Wang等人（2022年）研究了因果动态学习问题，该问题试图消除不相关的变量和行动与状态之间不必要的依赖。行动和状态变量之间不必要的依赖关系。

- Saengkyongam等人（2022年）。将因果性、不变性和上下文bandits联系起来。他们引入了政策不变性的概念，并表明在存在未观察到的变量的情况下，一个最佳的不变性政策会在不同的环境中具有普遍性。

- Ding等人（2022年）提出了一个新的解决方案，以解决目标条件强化学习（GCRL）中的泛化问题。学习（GCRL）中的泛化问题提出了一种新的解决方案，即把因果图视为一个潜在的变量，并使用变异的 可能性最大化的方法进行优化。这种方法训练智能体发现因果关系并学习一种 因果关系感知策略，该策略对不相关的变量具有鲁棒性。

**归纳到新的动态是一个更广泛的问题。它可能涉及物理属性的变化**（例如，重力加速度，如图5d所示）、模拟环境与现实的差异、属性值范围的变化等。

- Sontakke等人(2021)提议训练RL智能体对环境中的因果因素进行分类和推断，这将通过一种称为因果好奇心的创新内在动力来完成。智能体可以以自我监督的方式学习有语义的行为，学到的因果表征使他们有能力概括到未见的环境中。

- Lee等人(2021b)研究了如何通过进行干预来教机器人执行操纵任务，这有助于确定状态和行动空间之间的相关关系。在对相关特征进行领域随机化训练后，机器人表现出优秀的模拟到现实的泛化能力。

- Zhu等人(2021)开发了一种算法，以提高智能体对很少见或未见的物体属性的泛化能力。该算法使用单片机对环境动态进行建模，使智能体能够推理出如果物体有不同的属性值会发生什么，这导致了泛化能力的提高。

- Guo等人(2022)研究了无监督动态泛化问题，这使得模型能够泛化到新的环境中。作者遵循的直觉是，来自相同轨迹/类似环境的数据应该有类似的属性（隐藏变量），导致类似的因果效应。他们在调解分析中使用条件直接效应来衡量相似性。实验结果表明，所学模型在新的动态中表现良好。

#### 4.2.2 Generalize to Different Tasks

另一个重要的话题是如何对不同的任务进行归纳。在SCM框架中，不同的任务是通过改变奖励变量的结构方程或其在因果图上的父节点而产生的。这些任务具有相同的基础环境动态，但奖励的分配方式不同。

- Eghbal-zadeh等人(2021)引入了因果语境RL，其中智能体应该学习适应性策略，以适应由语境变量指定的新任务。他们提出了一个上下文注意模块，允许智能体将等分的特征作为上下文因素纳入，实现比非因果智能体更好的泛化。

- 为了使RL在复杂的多对象环境中更加有效，Pitis等人(2022)提出了局部因素。(2022)提出，应该识别和使用过渡动态中的局部因素。他们提出了一个称为基于模型的反事实数据增强的新框架，利用局部结构来产生反事实的过渡，使模型能够推广到OOD任务中。

此外，在现实中，泛化可能涉及环境动态和任务两方面的变化。一些研究从因果的角度探讨了这个问题。

- Zhang和Bareinboim(2017)使用因果推理来改善强化学习中的知识转移。他们的方法解决了当标准技术无法识别因果效应时，bandit智能体之间的知识转移问题。他们引入了一种新的识别策略，包括利用结构性知识推导出臂分布的界限，并将这些界限转移到其他地方，并将这些界限转移到新的bandit算法中。
- Dasgupta等人(2018)研究了元RL是否可以教智能体进行因果推理。实验结果显示，智能体成功地学会了进行干预，这支持了奖励准确因果推理的后续任务。智能体还学会了进行复杂的反事实预测，这种出现的能力可以有效地泛化到新的因果结构。
- Nair等人(2019)开发了一种使用有向无环图的方法，将因果知识传授给智能体。他们利用注意力机制，让智能体根据其视觉观察生成一个因果图，并利用它来做出明智的决定。实验表明，该智能体能有效地泛化到具有未知因果结构的新任务和环境。
- Huang等人(2022c)提出了AdaRL，这是一个自适应RL的框架，允许快速适应新环境、任务或观察。他们使用紧凑的图形表示法来编码决策问题中变量之间的结构关系，这使得只需几个样本就能有效地适应新领域的政策。

#### 4.2.3 Other Generalization Problems

在离线RL中，智能体只能从预先收集的数据集中学习。在这种情况下，智能体在测试阶段可能会遇到 在测试阶段，智能体可能会遇到以前未曾见过的状态-动作对，从而导致分布性转变问题(Levine et al., 2020)，这是离线RL的一个常见挑战。大多数现有的方法通过保守或悲观的学习来缓解这个问题(Fujimoto et al., 2019; Kumar et al., 2020; Yang et al., 2021b)，很少有考虑离线RL和泛化的结合(Kirk et al., 2022)。Zhu等人(2022b)提出了 提出了一个对未见过的状态进行概括的解决方案。他们通过使用因果发现技术从离线数据中恢复了因果结构。实验结果表明，因果世界模型比传统世界模型表现出更好的归纳性能。

### 4.3 Causal Reinforcement Learning for Addressing Spurious Correlations

正如我们在第3.3节中所说，RL智能体在了解环境和任务的过程中容易受到虚假关联的影响。根据决策问题背后的因果结构，虚假相关可以是两种类型之一：

一种是对应于由叉子结构引起的混杂性偏见，另一种是对应于由碰撞结构引起的选择性偏见。结构造成的混淆性偏差，另一种是由碰撞结构造成的选择性偏差（图6）。我们 将现有的方法相应地分为两类。特别是，我们还包括关于模仿学习(IL)和非策略评估(OPE)，因为它们都与RL的策略学习密切相关。代表性的作品见表3。

![table3](table3.png)

#### 4.3.1 Addressing Confounding Bias

我们首先介绍几种利用因果推理消除混杂偏差的技术(Glymour et al., 2016)。其中最常见的是后门调整方法，这种技术通过控制满足后门标准的变量来准确估计输入(treatment)和结果变量之间的因果效应。后门变量（如混杂因素）可以阻断所有虚假的路径，并确保因果关系充分说明输入和结果变量之间的相关性。如果不可能找到一组符合后门标准的协变量（例如，后门路径上的所有变量路径的所有变量都是不可观察的），我们可以使用前门调整方法。这种方法包括通过控制变量来估计通过控制从输入变量到结果变量的直接路径上的变量来估计因果效应。当这两种方法都不可行时，我们可以使用do-calculus(Pearl,1995)，这是一个完整的公理系统，它检索有助于识别因果效应的协变量，或在效应无法识别时报告失败。识别不了的情况下报告失败。这并不是一个解决混杂偏差的详尽技术清单，还有许多其他方法可以用来减轻混杂偏差。

MAB问题可以被认为是一个没有状态转换的单步决策问题。

- Forney等人(2017)研究了这个问题的一个变体，它涉及到影响行动和奖励的未测量的变量（未观察到的混杂因素）。他们发现，基于反事实的决策有助于解决这个问题，并促进观察和实验数据的融合。
- Zhang等人(2020b)使用示范数据和关于数据生成过程的定性知识的组合研究了单步模仿学习。他们提出了一个图形化的标准，用于确定在存在未观察到的混杂因素的情况下模仿的可行性，以及一个实用的程序，借助混杂的专家数据评估模仿策略。在随后的一篇论文中(Kumor et al., 2021)，这一方法随后被扩展到顺序性设置。
- Swamy等人(2022)设计了一种算法，用于有缺陷数据的模仿学习算法。他们提议使用工具变量回归(Stock & Trebbi, 2003)。一种著名的因果推理技术，来打破虚假的相关关系。

一些研究集中在离轨策略评价(OPE)问题上，它可以在政策部署前估计其性能。

- 例如，Namkoong等人(2020)通过制定评价策略性能的最坏性能情况界限，评估了OPE方法在未观察到的混杂情况下的稳健性。他们还提出了一个计算最坏情况界限的有效程序，允许可靠地选择策略。

- 另一方面，Bennett等人(2021)提出了一个新的估算器，用于估算具有未观察到的混杂因素情况下的无限跨度RL-OPE问题。他们专注于模型和策略价值可识别的特定环境。

  

- Lu & Lobato(2018)研究了因果推理和RL之间的联系。他们提出了一种方法，称为 deconfounding RL，它允许从受未观察因素影响的历史数据中学习好的策略。当应用于有混杂因素的观察性数据时，这种方法优于传统的RL方法。

- Liao等人(2021)也专注于离线设置。他们发现，未观察到的混杂因素通常影响观察研究中的行动。他们提出了一种算法，帮助有效识别 在RL中使用工具变量的过渡动态。

- Wang等人(2021b)则提出了 提出了一种将离线数据纳入在线环境的方法，考虑到可能影响数据准确性的混杂变量。影响数据准确性的混杂变量。这种方法有效地调整了混杂的偏差，并取得了比最佳的在线数据更小的遗憾(regret)。

- Gasse等人(2021)研究了一种基于模型的方法，该方法结合干预性和观察性数据来学习一个基于隐式(latent-based)的因果转换模型，然后用它通过deconfounding来解决POMDP。

- Yang等人(2022a)提出了一种称为因果推断Q-network的算法，以处理多种类型的观察性干扰引起的混杂偏差。由于因果推理在去伪存真上的优势，这种方法抗干扰更强，性能更好。

- Rezende等人(2020)讨论了RL中部分模型的使用，这是一种基于模型的方法，不需要对全部（通常是高维）观察进行建模。当部分模型被它们没有建模的部分所混淆时，就会出现问题，导致不正确的规划。作者提出了一个解决方案，通过确保部分模型的因果准确性来克服这一缺陷。

#### 4.3.2 Addressing Selection Bias

当数据样本不能正确代表目标人群时，就会出现选择性偏差。

>  For example, selective bias arises when researchers seek to understand the effect of a certain drug on curing a disease by investigating patients in a selected hospital. This is because those patients may differ significantly from the population regarding where they reside, their social status, and their wealth, making them unrepresentative.

- Bai等人(2021)研究了在目标条件强化学习(GCRL)问题中使用事后经验重放(HER)所带来的选择性偏差。特别是，HER重新标记了每个收集的轨迹的目标，允许智能体从失败中学习（没有达到原来的目标）。令人担忧的是，重新标记的目标分布不能正确代表原始目标分布；因此，用HER训练的智能体是有偏见的。作者提出在因果推理中使用反概率加权技术进行学习，这使得智能体在HER的帮助下提高了样本效率，同时避免了因重新标记而产生的偏差，在一系列机器人操纵任务上取得了可喜的成果。

- Deng等人（2021）通过选择性偏见的视角看待离线RL问题。智能体在离线设置中容易受到不确定性和决策之间的虚假相关性的影响，容易学习次优策略。通过因果模型，我们可以看到，经验回报是不确定性和实际回报综合的结果。由于在离线环境中不可能通过获取更多的数据来消除不确定性，一个没有因果意识的智能体可能会错误地认为不确定性和收益之间存在着不确定性和收益之间的因果关系。因此，它更喜欢那些通过偶然（高不确定性）获得高收益的策略。作者建议量化不确定性，并将其作为学习过程中的一个惩罚项在学习过程中使用。结果表明，这种方法优于各种不考虑因果关系的各类离线RL基线。

### 4.4 Beyond Return with Causal Reinforcement Learning

随着基于RL的自动化决策系统在人类社会的各种活动中得到广泛应用，我们将对这些系统产生各种不同的关心。在本节中，我们将讨论如何通过因果建模来解决这些问题。代表作品见表4。

![table4](table4.png)

#### 4.4.1 Explainable Reinforcement Learning via Causal Modeling

一个好的RL智能体应该能够成功地解决任务，并解释他们的行为。大型语言模型为因果关系增强交流提供了极好的例子。人类在日常生活中使用因果语言，因此在大规模人类书写文本上训练的大型语言模型可以自然流畅地与人类交流。一般来说，使用SCM框架对数据生成过程建模的因果RL算法本质上是可解释的。我们可以在训练之前向智能体提供先验知识(例如因果图)，确保智能体和人类对环境有相同的理解。使用因果建模，智能体学习因果关系而不是相关性。因此，我们可以在因果层面上理解智能体的决策，而不是依赖于模糊的相关性。

在实践中，我们常常渴望包含反事实的细粒度解释。“反事实”一词在多智能体强化学习(MARL)中很流行。

- 例如，Foerster等人(2018)提出了一种名为反事实多智能体策略梯度的方法，用于在合作多智能体系统中有效学习分散策略。更准确地说，反事实解决了多智能体共享分配(Credit Assignment)的挑战，使智能体和人类能够更好地理解个人行为对团队的贡献。
- 随后的一些研究也遵循了同样的观点(Su et al., 2020;Zhou et al., 2022)。这些方法没有执行如图7所示的完整的反事实推理过程，缺少了关键的诱导(abduction)步骤，这为进一步增强提供了机会。
- 最近，Triantafyllou等人(2022)在Dec-POMDPs和SCM之间建立了联系，使他们能够使用因果语言研究MARL中的贡献分配问题。然而，这项工作并不是为了提高RL智能体的性能；相反，它的目标是引入实际因果关系的正式定义，以建立支持问责制的RL框架，这是负责任决策的关键。
- 另一方面，Mesnard等人(2021)研究了时间共享分配问题，这有助于理解特定行为对未来奖励的影响。



- Madumal等人(2020)使用认知科学的理论来解释人类如何通过因果关系来理解世界，以及这些关系如何帮助我们理解和解释RL智能体行为。他们提出了一种在RL过程中学习SCM的方法，并使用该模型生成基于反事实分析的行为解释。他们对120名参与者进行了一项研究。结果表明，基于因果关系的解释在理解、解释满意度和信任方面优于其他解释模型。
- Bica等人(2021a)讨论了一种通过将反事实推理集成到批量逆向RL中来理解和解释专家决策过程的方法。通过使用反事实，可以解释专家行为，并在禁止积极实验的情况下评估策略。
- Tsirtsis等人(2021)研究了如何为序列决策过程找到最佳的反事实解释。他们将这个问题定义为一个有约束的搜索问题，即如何通过指定数量的动作来搜索与观察到的序列不同的动作序列。
- Herlau & Larsen(2022)研究了RL中的中介分析技术。特别是，他们迫使RL智能体在学习过程中最大化自然间接效应，这使得智能体能够识别决策问题中的关键事件，例如在打开一扇门之前获得钥匙。通过中介分析，智能体可以学习到一个简约的描述性因果模型，这为可解释的RL提供了一个新的视角。

#### 4.4.2 Fair Reinforcement Leaning via Causal Modeling

除了可解释性，我们还希望RL代理与人类价值观保持一致，避免对人类社会造成潜在危害。公平是一个重要的考虑因素。然而，关于现实生活中的公平性的研究很少。

- 一个简单的想法是使用统计测量来量化公平，并将其视为约束(Balakrishnan等人，2022)。然后，策略优化就变成了一个有约束的优化问题，可以使用诸如Gurobi 5这样的强大求解器来解决。然而，公平性提出了一个反事实的问题：如果敏感属性的值不同，结果会不同吗?为了准确地评估公平性，人们必须评估与事实相反的数量。
- Zhang & Bareinboim(2018)首次引入SCM框架来阐述公平的概念，使研究人员能够定量地评估反事实的公平性。使用反事实陈述，研究人员可以系统地分析不同类型的歧视及其对决策的影响。
- Huang等人(2022c)研究了推荐场景中的公平性，重点是bandit设置(单步决策)，其中敏感属性不应影响奖励。
- Liu et al.(2020)也从因果关系的角度研究了RL中的公平问题。他们采用因果图来提供公平问题的形式化分析，并使用反事实来评估公平。实验结果表明，该方法可以在多种问题设置下学习公平策略。

#### 4.4.3 Safe Reinforcement Learning via Causal Modeling

最后，安全是让RL在现实世界中广泛应用的一个根本挑战。因果模型为研究安全性提供了一些有价值的工具。

- 例如，Hart & Knoll(2020)调查了与自动驾驶相关的安全问题。研究人员可以利用反事实推理，在将政策部署到现实世界之前进行反事实的政策评估。实验结果表明，他们的方法具有很高的成功率，同时显著降低了碰撞率。
- 另一方面，Everitt等人(2021)研究了足够有能力的RL代理的奖励篡改问题。这些代理可能会寻找捷径来获得奖励，而不是执行预期的行为，从而带来潜在的安全风险。作者使用因果图提出了这个问题的精确和直观的形式化，并提出了防止奖励篡改的设计原则。

## 5 Open Problems and Future Directions

这一节我们将考虑在因果RL中许多重要的还未经探索的话题。

### 5.1 Causal Learning in Reinforcement Learning

在前一节中，我们解释了因果动力学学习(与MBRL密切相关的一类方法)如何提高样本效率和可泛化性(Wang et al., 2022; Huang et al., 2022b)。这些方法侧重于理解变量之间的因果关系以及产生这些变量的过程。这些方法不使用复杂的冗余连接来对数据生成过程建模，而是采用稀疏的模块化风格。因此，它们比传统的基于模型的方法更有效和稳定，并允许RL智能体快速适应看不见的环境或任务。然而，在现实中，我们对因果变量的先验知识可能并不完全。有时，我们必须处理高维和非结构化数据，如视觉信息。在这种情况下，RL智能体需要能够从原始数据中提取因果表示(Schölkopf et al., 2021)。根据任务的不同，因果表征可以是抽象的概念，如情绪和偏好，也可以是更具体的东西，如物理对象。

![table5](table5.png)

从数据中学习因果模型的完整过程被称为因果学习(Peters et al., 2017)。它不同于因果推理(Imbens & Rubin, 2015; Glymour et al., 2016)，它只关注在给定的因果模型下估计特定的因果效应。因果学习包括提取因果特征，发现因果关系，学习因果机制。表5简要总结了它们的特点。

这三个因素都很重要，值得进一步研究。大量关于因果发现的研究已经完成(Spirtes et al., 2000; Pearl, 2009 b; Peters et al., 2017; Vowels et al., 2022)，从数据中恢复一组变量的因果结构的过程，特别是关于条件独立性检验(Spirtes et al., 2000; Sun et al.， 2007; Hoyer et al., 2008; Zhang et al., 2011)。在某些假设下，例如可信度，算法可以从观测数据中识别潜在因果图的马尔可夫等价类。将因果发现与RL相结合，可以让智能体以交互的方式主动地从环境中收集干预数据。因此，该领域的一个有趣的研究方向是如何使用干预数据或观察数据和干预数据的组合来有效地发现因果关系(Addanki et al., 2020; Jaber et al., 2020; Brouillard et al., 2020; Zhu et al., 2022a)。
