import torch
import random

# Value(2.0)  torch.tensor(2.0,requires_grad=True)   建一个带梯度的节点
# .data       .item()/张量本身                        原本只能存一个数，现在能存整个矩阵
# .grad       .grad                                  完全一样，backward()后填充
# .grad=0.0初始化    optimizer.zero_grad()        梯度必须手动清零
# o.backward()      loss.backward()              一样，只是起点变成loss
# __add__/__mul__/tanh      2000+个算子          原本只有几个，它有几千个
# 每个运算里的_backward闭包    C++实现的autograd引擎    同一套链式法则
# build_topo+visited   内部动态图        看不见，但机制一模一样
# 手写更新w.data-=lr*w.grad     torch.optim.SGD/Adam   优化器帮你管
# 无           nn.Moudle\DataLoader\GPU\分布式     工程层的东西

x1=torch.Tensor([2.0]).double(); x1.requires_grad=True
x2=torch.Tensor([0.0]).double(); x2.requires_grad=True
w1=torch.Tensor([-3.0]).double(); w1.requires_grad=True
w2=torch.Tensor([1.0]).double(); w2.requires_grad=True
b=torch.Tensor([6.8813735870195432]).double(); b.requires_grad=True
#tensor有什么意义：存储数据，计算图，梯度
#和Value()方法的区别是？Value()方法只存储数据，tensor存储数据，计算图，梯度
n=x1*w1+x2*w2+b
o=torch.tanh(n)

print(o.data.item)
#o.item()和o.data.item()在Pytorch里是一样的
#o.data.item()  #tensor的值   仅仅是o.data是什么意思？存储数据，计算图，梯度，这里只是取值
#o.grad         #tensor的梯度
#o.backward()   #计算梯度
#o.grad=0.0     #初始化梯度
#o.grad.data    #tensor的梯度
#o.grad.data.zero_()  #初始化梯度
#为什么用item不用data?因为item只能存一个数，data能存整个矩阵
o.backward()
print("______")
print('x2',x2.grad.item())
print('w2',w2.grad.item())
print('x1',x1.grad.item())
print('w1',w1.grad.item())

torch.Tensor([2.0]).double().dtype
#tensor的dtype
#张量是什么？所以是一种数据类型吗？不是，它是一种数据结构，存储数据，计算图，梯度。
#感觉和Value()方法差不多，只不过Value()方法只存储数据，tensor存储数据，计算图，梯度。
#Value()方法是什么？Value()方法是一种数据类型，存储数据，计算图，梯度。

class Value:
    #定义一个名为Value的类，用于表示一个数值对象。该类具有一个属性data，用于存储数值数据。
    #__init__方法是类的构造函数，用于初始化对象的属性。它接受data参数作为数值数据，并将其赋值给对象的data属性。
    #_children参数用于指定与该数值对象相关的子对象，并将其存储在_prev属性中。
    #_op参数用于指定与该数值对象相关的操作类型，并将其存储在_op属性中。
    def __init__(self,data,_children=(),_op='',label=""):
        self.data=data
        self._prev =set(_children)
        self._op=_op
        self.label=label
        self.grad=0.0
    #represent方法用于定义对象的字符串表示形式，当使用print函数打印对象时，会调用该方法返回一个可读的字符串。
    def __repr__(self):
        return f"Value(data={self.data})"
    #为什么方法一定要加下划线？
    # 因为在Python中，双下划线开头和结尾的方法（如__init__和__repr__）是特殊方法，用于定义对象的行为和属性。
    # 它们是Python的内置方法，用于实现对象的初始化、字符串表示等功能。通过使用这些特殊方法，可以自定义类的行为，
    # 使其更符合特定需求。
    #不加下划线和加了下划线有什么区别？
    # 而不加下划线的方法是普通方法，用于实现类的功能逻辑，不具有特殊的行为和属性。
    #other是另一个Value对象，表示要与当前对象进行加法运算的数值。
    def __add__(self,other):
        out =Value(self.data+other.data,(self,other),'+')
        return out
    def __mul__(self,other):
        out =Value(self.data*other.data,(self,other),'*')
        return out 

#神经网络之间的关系是什么：神经元，神经网络层，多层感知机
#神经元是什么：神经元是神经网络的基本单元，它接受输入，通过加权求和和激活函数，输出一个值。
#神经网络层是什么：神经网络层是由多个神经元组成的，它们接受输入，通过加权求和和激活函数，输出一个值。
#多层感知机是什么：多层感知机是由多个神经网络层组成的，它们接受输入，通过加权求和和激活函数，输出一个值。
#神经网络之间的关系是什么：神经元是神经网络的基本单元，神经网络层是由多个神经元组成的，多层感知机是由多个神经网络层组成的。
#神经网络输出的结果是什么：神经网络输出的结果是经过多个神经网络层计算后的结果。
#结果有什么意义：结果的意义取决于具体的任务，例如，在分类任务中，结果可以是每个类别的概率分布；在回归任务中，结果可以是连续的数值。


class Neuron:
    #定义一个名为Neuron的类，用于表示一个神经元对象。该类具有一个属性w，用于存储神经元的权重。
    def __init__(self,nin):#nin是输入的个数
        self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]#随机生成nin个权重,_是占位符，表示不需要使用这个变量
        #权重为什么可以随机生成？因为权重是随机初始化的，所以需要随机生成。
        self.b=Value(random.uniform(-1,1))#随机生成一个偏置

    #__call__方法用于定义对象的调用行为，当对象被当作函数调用时，会调用该方法。
    def __call__(self,x):
        #x是输入的值，self.w是权重，self.b是偏置
        # x*w+b  点积是什么意思？是两个向量的乘积，这里x和w都是向量，所以是点积。
        zip(self.w,x)#zip函数用于将两个列表中的元素一一对应，这里是将self.w和x中的元素一一对应。
        #zip函数返回的是一个zip对象，需要使用list函数将其转换为列表。
        print(list(zip(self.w,x)))
        act=sum(( wi*xi  for wi,xi in zip(self.w,x)),self.b) #sum函数用于计算列表中所有元素的和，这里是将self.w和x中的元素一一对应，然后计算它们的乘积，最后计算它们的和。
        out=act.tanh() #tanh函数用于计算tanh激活函数，这里是将act作为输入，计算tanh激活函数的输出。
        return out

    def parameters(self):#定义一个名为parameters的函数，用于返回神经元的权重和偏置。
        #return self.w+[self.b]#返回神经元的权重和偏置，这里是将self.w和self.b合并为一个列表，然后返回这个列表。
        return self.w+[self.b]#这里不是添加权重和偏置了吗？为什么还要返回self.w+[self.b]？因为self.w是神经元的权重，self.b是神经元的偏置，所以需要返回self.w+[self.b]。
    #那为什么layer那一层还要再保存一遍，不是多此一举吗？不是，因为layer那一层需要保存神经元的权重和偏置，而neuron那一层只需要保存神经元的权重和偏置，所以需要保存两次。

class Layer:#定义一个名为Layer的类，用于表示一个神经网络层对象。该类具有一个属性neurons，用于存储神经元的列表。
    def __init__(self,nin,nout):#nin是输入的个数，nout是输出的个数
        self.neurons=[Neuron(nin) for _ in range(nout)]

    #__call__方法用于定义对象的调用行为，当对象被当作函数调用时，会调用该方法。
    def __call__(self,x):
        outs=[n(x) for n in self.neurons]#将输入x传递给每个神经元，并返回神经元的输出列表。
        return outs[0] if len(outs)==1 else outs#将输入x传递给每个神经元，并返回神经元的输出列表。

    def parameters(self):#定义一个名为parameters的函数，用于返回神经元的权重和偏置。
        # params=[]#定义一个空列表，用于存储神经元的权重和偏置。
        # for neuron in self.neurons:#遍历神经元的列表。
        #     ps=neuron.parameters()
        #     params.extend(ps)#将神经元的权重和偏置添加到params列表中。
        # return params#我想知道这一步和neurons里的有什么区别：这一步是将神经元的权重和偏置添加到params列表中，而neurons里的只是存储神经元的列表，并没有添加权重和偏置。
        return [p for neuron in self.neurons for p in neuron.parameters()]#将神经元的权重和偏置添加到params列表中。


class MLP:#定义一个名为MLP的类，用于表示一个多层感知机对象。该类具有一个属性layers，用于存储神经网络的层列表。
    def __init__(self,nin,nouts):
        sz=[nin]+nouts #将输入的个数和输出的个数列表合并为一个列表。
        self.layers=[Layer(sz[i],sz[i+1]) for i in range(len(nouts))]#将每个层都初始化为一个Layer对象，并将它们存储在self.layers中。

    def __call__(self,x):
        for layer in self.layers:
            x=layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layer for p in layer.parameters()]#将神经元的权重和偏置添加到params列表中。

x=[2.0,3.0,-1.0]
n=MLP(3,[4,4,1])#输入的什么意思？输入的个数，输出的个数
#[4,4,1]是什么意思？表示神经网络的层数和每层的神经元个数。这不是三个数字吗？是的，这里表示神经网络的层数和每层的神经元个数。例如，[4,4,1]表示神经网络的层数为3，第一层有4个神经元，第二层有4个神经元，第三层有1个神经元。
#每个神经元用来做什么的？每个神经元用于计算输入的加权和，并通过激活函数输出一个值。
print(n(x))

xs=[
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]
]#输入是什么意思？输入是神经网络的输入数据，这里表示神经网络的输入数据为[2.0,3.0,-1.0]，[3.0,-1.0,0.5]，[0.5,1.0,1.0]，[1.0,1.0,-1.0]。
ys=[1.0,-1.0,-1.0,1.0]#输出是什么意思？输出是神经网络的预测结果，这里表示神经网络的预测结果为1.0，-1.0，-1.0，1.0。
ypred=[n(x) for x in xs]#ypred是什么意思？ypred是神经网络的预测结果，这里表示神经网络的预测结果为[n(x) for x in xs]。
#为什么会预测，预测的是什么？预测是指神经网络根据输入数据，通过计算得到输出数据的过程。预测的结果是神经网络的输出数据，这里表示神经网络的输出数据为[n(x) for x in xs]。
#权重不是随机的吗？到底在预测什么？权重是随机初始化的，但是通过训练，神经网络的权重会逐渐调整，使得预测的结果更加准确。预测的结果是神经网络的输出数据，这里表示神经网络的输出数据为[n(x) for x in xs]。
print(ypred)


loss=sum((yout-ygt) for ygt,yout in zip(ys,ypred))#真值与预测值对应
    
loss.backward()
n.layers[0].neurons[0].w.grad  
#查看什么梯度？查看神经元的权重梯度。
# 为什么是0？因为神经元的权重梯度是第一个神经元的权重梯度。
# 那其他神经元的权重梯度呢？其他神经元的权重梯度可以通过n.layers[0].neurons[1].w.grad等方式查看。
# 为什么这里只看第一个？是为了方便查看。

for p in n.parameters():
    p.data+=-0.01*p.grad#更新神经元的权重，这里是将神经元的权重减去0.01倍的权重梯度，得到新的权重。


for k in range(10):
    #forward pass
    ypred=[n(x) for x in xs]
    loss=sum((yout-ygt) for ygt,yout in zip(ys,ypred))
    
    #backward pass
    for p in n.parameters():
        p.grad=0.0
    loss.backward()
    #update weights
    for p in n.parameters():
        p.data+=-0.01*p.grad
            # #zero gradients
            # for p in n.parameters():
            #     p.grad=0.0
#为什么这里要zero gradients？因为梯度是累积的，如果不清零，梯度会一直累积，导致权重更新不准确。
#为什么这里要更新权重？因为神经网络的权重是随机初始化的，通过训练，神经网络的权重会逐渐调整，使得预测的结果更加准确。

# Common errors:

# - you didn't try to overfit a single batch first.#你没有尝试首先过拟合一个批次。
# - you forgot to toggle train/eval mode for the net.#你忘记了切换网络的训练/评估模式。
# - you forgot to .zero_grad() (in pytorch) before .backward().#你忘记了在.backward()之前.zero_grad()（在pytorch中）。
# - you passed softmaxed outputs to a loss that expects raw logits.#你将softmaxed输出传递给了期望原始logits的损失。
# - others? :)#还有其他吗？:)