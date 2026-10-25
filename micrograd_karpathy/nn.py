import random
from engine import Value

# Moudle 是什么？
# Moudle 是一个容器，它包含了一组参数，这些参数可以是一个 Value 对象，也可以是一个 Moudle 对象。Moudle 对象可以包含多个子 Moudle 对象，形成一个树状结构。
class Module:
    #zero_grad() 方法的作用是什么？
    #zero_grad() 方法的作用是将所有参数的梯度清零。在反向传播过程中，梯度会累加，如果不清零，梯度会不断累加，导致结果不正确。
    def zero_grad(self):
        for p in self.parameters():
            p.grad=0
    def parameters(self):
        return []

#Neuron 是什么？
#Neuron 是一个神经元，它包含了一组权重和偏置，以及一个激活函数。Neuron 对象可以包含多个子 Neuron 对象，形成一个树状结构。
class Neuron(Module):
    def __init__(self,nin,nonlin=True):
        # nin 是输入的维度
        # nonlin 是是否使用激活函数
        self.w=[Value(random.uniform(-1,1))for _ in range(nin)]
        self.b=Value(0)
        self.nonlin=nonlin

    def __call__(self,x):
        #__call__ 方法的作用是什么？
        #__call__ 方法的作用是定义对象被调用时的行为。
        # 在这个例子中，Neuron 对象被调用时，会计算输入 x 和权重 w 的乘积，再加上偏置 b，然后返回结果。
        #zip() 函数的作用是什么？
        #zip() 函数的作用是将两个可迭代对象（如列表、元组等）中的元素一一对应，返回一个元组的列表。在这个例子中，zip() 函数将权重 w 和输入 x 中的元素一一对应，然后计算它们的乘积。
        act=sum((xi*wi for wi,xi in zip(self.w,x)),self.b)
        return act.reLU() if self.nonlin else act
        #这里在做什么？
        #这里在判断是否使用激活函数。如果 self.nonlin 为 True，则使用 ReLU 激活函数，否则直接返回结果。
    
    def parameters(self):
        # parameters() 方法的作用是什么？
        #parameters() 方法的作用是返回所有参数。在这个例子中，Neuron 对象的参数包括权重 w 和偏置 b。
        return self.w+[self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    #__repr__ 方法的作用是什么？
    #__repr__ 方法的作用是返回对象的字符串表示。
    # 在这个例子中，Neuron 对象的字符串表示是 "ReLU" 或 "Linear" 加上权重 w 的长度。

class Layer(Module):
    def __init__(self,nin,nout,**kwargs):
        # nin 是输入的维度
        # nout 是输出的维度
        # **kwargs 是其他参数
        #两个星号啥意思？
        #两个星号表示将字典中的键值对作为参数传递给函数。在这个例子中，**kwargs 将字典中的键值对作为参数传递给 Neuron 对象的构造函数。
        self.neurons=[Neuron(nin,**kwargs) for _ in range(nout)]
        #这里在做什么？
        #这里在创建一个包含 nout 个 Neuron 对象的列表。每个 Neuron 对象的输入维度为 nin，其他参数由 **kwargs 提供。

    def __call__(self,x):
        out=[n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out
    # 这里在做什么？
    # 这里在调用每个 Neuron 对象，并将输入 x 传递给它。如果只有一个 Neuron 对象，则返回它的输出，否则返回一个包含所有 Neuron 对象输出的列表。

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    # 这里在做什么？
    # 这里在返回一个包含所有 Neuron 对象参数的列表。

    def __repr__(self):
        return f"Layer of [{','.join(str(n) for n in self.neurons)}]"
    # 这里在做什么？
    # 这里在返回一个字符串，表示 Layer 对象的字符串表示。

class MLP(Module):
    def __init__(self,nin,nouts):
        sz=[nin]+nouts
        self.layers=[Layer(sz[i],sz[i+1],nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        # 这里在做什么？
        # 这里在创建一个包含多个 Layer 对象的列表。每个 Layer 对象的输入维度为 sz[i]，输出维度为 sz[i+1]，最后一个 Layer 对象不使用激活函数。
        #nonlin=i!=len(nouts)-1 是什么意思？
        #nonlin 是一个布尔值，表示是否使用激活函数。如果 i 不等于 len(nouts)-1，则使用激活函数，否则不使用激活函数。

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x
    # 这里在做什么？
    # 这里在调用每个 Layer 对象，并将输入 x 传递给它。最后返回最后一个 Layer 对象的输出。

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    # 这里在做什么？
    # 这里在返回一个包含所有 Layer 对象参数的列表。
    def __repr__(self):
        return f"MLP of [{','.join(str(layer) for layer in self.layers)}]"
    # 这里在做什么？
    # 这里在返回一个字符串，表示 MLP 对象的字符串表示。


#Moudle(基类)->Neuron(神经元)-Layer(层)-MLP(多层感知机)

#Module:parameters() 返回参数列表（基类返回空，等子类去重写）
        # zero_grad() 把所有参数梯度清零
#Neuron: __call__() 让对象能像函数一样被调用：n(x)等价于n.__call__(x)
        #zip(self.w,x)->权重和输入一一对应
        #wi*xi->逐个相乘（每个都是Value，会建图）
        #sum(...,self.b)->全部相加，起点是偏置b
        #.relu()->过激活函数（如果需要）
        #parameters()->参数持有者，self.w+[self.b]把权重列表和偏置拼成一个新列表

#layer 一层神经元：Layer 不持有参数，只做汇总
        #**kwargs 两个星号表示“收集所有关键字参数打包成字典，再原样转发”
        #透传：Layer(3, 4, nonlin=False) → 字典 {'nonlin': False} → 转发给每个 Neuron(3, nonlin=False)。
        # 这样 Layer 自己根本不需要知道 Neuron 有哪些参数，加新参数时 Layer 一行都不用改
        #Layer.__call__()→逐个调用每个神经元，并收集输出，对 nout 个神经元各跑一遍。那个 out[0] if len(out)==1 是 Python 的"标量退化"技巧：
        # 输出层只有 1 个神经元时，直接返回那个 Value 而不是长度为 1 的列表
        # def parameters(self):                         
            # return [p for n in self.neurons for p in n.parameters()]
    # 等价于
        # params = []
        # for n in self.neurons:        # 遍历每个神经元
        #     for p in n.parameters():  # 取它的参数
        #         params.append(p)      # 摊平

#MLP 多层感知机：MLP 不持有参数，只做汇总
        #MLP.__call__()→逐个调用每个 Layer 对象，并收集输出
        # sz = [3,4,4,1] 这种写法叫"尺寸链"，相邻两个数字就是一层的 (输入, 输出)：
        # 3→4、4→4、4→1。
        # nonlin = i != len(nouts)-1​ 是关键设计：
        # i=0: 0 != 2 → True → 用 ReLU
        # i=1: 1 != 2 → True → 用 ReLU
        # i=2: 2 != 2 → False → 最后一层不加激活
        # 因为回归任务的输出不该被 ReLU 截断（负值会被压成 0）。
        # x = layer(x) 这个自我覆盖就是"前向传播"的全部——数据逐层流过。



# 结构: MLP(3, [4,4,1])

# MLP(3, [4, 4, 1])   →  尺寸链 sz = [3, 4, 4, 1]

# 第0层: 3 → 4    权重 12 个 + 偏置 4 个 = 16 个参数   [ReLU]
# 第1层: 4 → 4    权重 16 个 + 偏置 4 个 = 20 个参数   [ReLU]
# 第2层: 4 → 1    权重 4 个  + 偏置 1 个 = 5 个参数    [Linear 无激活]
#                                     合计参数量: 41
# 输入 len=3  →  输出一个标量 Value
#   第0层: Layer of [ReLUNeuron(3) ×4]
#   第1层: Layer of [ReLUNeuron(4) ×4]
#   第2层: Layer of [LinearNeuron(4)]     ← 最后一层确实是 Linear ✅

# 参数量: 16 + 20 + 5 = 41   (代码给出 41 ✅)

# 训练 60 步 (SGD, lr=0.05):
#   step  0: loss = 8.7172
#   step 15: loss = 1.6315
#   step 30: loss = 0.0657
#   step 59: loss = 0.2143

# 最终预测 vs 目标:
#   [2, 3, -1]    → +0.3817  (目标 +1.0)
#   [3, -1, 0.5]  → -0.9971  (目标 -1.0) ✅
#   [0.5, 1, 1]   → -0.9971  (目标 -1.0) ✅
#   [-1, -1, 0.5] → +0.9055  (目标 +1.0) ✅


# Module (基类)
#  ├── parameters()  → []          给子类重写
#  └── zero_grad()   → 依赖 parameters()，一次写好全网络通用
#        │
#        ├── Neuron  【持有者】w 和 b 存在这里，唯一真正存参数的地方
#        │    └── parameters() = self.w + [self.b]
#        │
#        ├── Layer   【汇总者】nout 个 Neuron 平铺
#        │    └── parameters() = 摊平每个 neuron.parameters()
#        │
#        └── MLP     【总汇总者】多个 Layer 串联
#             └── parameters() = 摊平每个 layer.parameters()