from numpy import arange
import numpy as np
import math
# 绘制函数图像
#graphviz是一个用于创建图形和可视化的库，提供了简单的接口来生成有向图和无向图。
#digraph是graphviz库中的一个类，用于创建和操作有向图。
# 它提供了添加节点、添加边、设置节点和边的属性等功能，可以用于表示和分析图结构。
from graphviz import Digraph

def trace(root):
    # 创建一个有向图对象
    nodes,edges=set(),set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child,v))
                build(child)
    build(root)
    return nodes,edges

def draw_dot(root):
    # Digraph是graphviz库中的一个类，用于创建和操作有向图。
    #formast='svg'指定生成的图像格式为SVG（可缩放矢量图形）。
    #graph_attr={'rankdir':'LR'}设置图的属性，其中rankdir='LR'表示图的布局方向为从左到右。
    dot=Digraph(format='svg',graph_attr={'rankdir':'LR'})
    #root是一个Value对象，表示计算图的根节点。通过调用trace函数，获取计算图中的所有节点和边。
    #所以是root中含有所有的节点和边，trace函数会递归地遍历计算图，将所有节点和边添加到nodes和edges集合中。
    nodes,edges=trace(root)
    for n in nodes:
        #id(n)返回对象n的唯一标识符（内存地址），将其转换为字符串作为节点的唯一标识符。
        #对于图中的任何值，为其创建一个矩形（'record'）节点。
        uid =str(id(n))
        #label="{data %.4f}"%(n.data,)意思是将节点的标签设置为"data "后跟n.data的值，保留四位小数。
        # 通过使用字符串格式化，将n.data的值插入到标签中。
        #shape="record"表示节点的形状为矩形（记录形状）。
        dot.node(name=uid,label="{date %.4f}"%(n.data,),shape="record")
        if n._op:
            #如果节点n有操作类型（_op不为空），则为该操作创建一个圆形节点，并将其与节点n连接起来。
            dot.node(name=uid+n._op,label=n._op)
            #并且将操作节点与节点n连接起来，形成有向边。
            dot.edge(uid+n._op,uid)

    for n1,n2 in edges:
        #连接所有的边，将节点n1和节点n2连接起来，形成有向边。
        dot.edge(str(id(n1)),str(id(n2))+n2._op)

    return dot

class Value:
    #定义一个名为Value的类，用于表示一个数值对象。该类具有一个属性data，用于存储数值数据。
    #__init__方法是类的构造函数，用于初始化对象的属性。它接受data参数作为数值数据，并将其赋值给对象的data属性。
    #_children参数用于指定与该数值对象相关的子对象，并将其存储在_prev属性中。
    #_op参数用于指定与该数值对象相关的操作类型，并将其存储在_op属性中。
    #a+b a.__add__b(b)
    #a*b a.__mul__b(b)
    #a/b a.__truediv__(b)
    #a**2 a.__pow__(2)
    #2**a a.__rmul__(2) (反射)
    #len(a) a.__len__()
    #a[0] a.__getitem__(0)

    def __init__(self,data,_children=(),_op='',label=""):
        self.data=data
        self.grad=0.0
        self.backward=lambda:None
        #lambda的意思是返回一个空值None
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
        def _backward():
            self.grad+=1.0*out.grad
            other.grad+=1.0*out.grad
        out._backward=_backward
        return out

    def __mul__(self,other):
        out =Value(self.data*other.data,(self,other),'*')
        def _backword():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backword=_backword
        return out 

    def __pow__(self,other):
        assert isinstance(other,(int,float)),"only supporting int/float powers for now"
        out=Value(self.data**other,(self,),f'**{other}')
        def _backward():
            self.grad+=other.data*out.grad#exercise
        out._backward=_backward

    
    def __rmul__(self,other):#other * self
        return self*other
    

    def __truediv__(self,other):#self/other
        return self*other**-1

    def __neg__(self):
        return self*-1
    
    def __sub__(self,other):
        return self+(-other)

    def tanh(self):
        x=self.data
        t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
        out=Value(t,(self,),'tanh')
        #value(操作对象，操作类型，操作名称)
        def _backword():
            self.grad+=(1-t**2)*out.grad
        out._backward=_backword
        return out

    def exp(self):
        x=self.data
        out=Value(math.exp(x),(self,),'exp')
        def _backward():
            self.gard+=out.data*out.grad
        out._backward = _backward

        return out

    def backward(self):
        o.grad=1.0
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(o)
        for node in reversed(topo):
            node._backword()


#label的作用是什么？
#label是给节点添加标签，用于在图形中显示节点的名称。
#label的值是字符串，用于描述节点的功能和属性。
#inputs x1,x2
#inputs是输入，这里是x1,x2
x1=Value(2.0,label="x1")
x2=Value(0.0,label="x2")
#weihghts w1,w2
#weights是权重，这里是w1,w2
w1=Value(-3.0,label='w1')
w2=Value(1.0,label='w2')
#bias of the neuron
#neuron意思是神经元，这里是偏置项
b=Value(6.7,label="b")
#x1*w1 +x2*w2 +b
x1w1=x1*w1; x1w1.label="x1*w1"
x2w2=x2*w2; x2w2.label="x2*w2"
x1w1x2w2=x1w1+x2w2;x1w1x2w2.label="x1*w1+x2*w2"
n=x1w1x2w2+b;n.label="n"
o=n.tanh();o.label="o"

#tanh()是激活函数，怎么激活的？
#tanh()是双曲正切函数，用于将n的值映射到[-1,1]之间。
#backward()是反向传播函数，用于计算n的梯度。
#relu()是激活函数，用于将n的值映射到[0,inf]之间。
# o=n.tanh()
print(draw_dot(n))

o.backward()
o.grad=1.0

topo=[]
visited=set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
build_topo(o)
for node in reversed(topo):
    node._backword()


#如果变量使用超过一次会出现错误，比如b=a+a,对a的grad求解会出现问题
#所以改为+=，这样同一变量的所有贡献会相加





o.grad=1.0
o._backward()
n._backward()
b._backward()
x1w1x2w2._backward()
x2w2._backward()
x1w1._backward()

# x1.grad=w1.data*x1w1.grad
# w1.grad=x1.data*x1w1.grad

# x2.grad=w2.data*x2w2.grad
# w2.grad=x2.data*x2w2.grad

# x1w1.grad=0.5
# x2w2.grad=0.5

# x1w1x2w2.grad=0.5
# b.grad=0.5

# b.grad=0.5
# o.grad=1
# #o=tanh(n)
# #do/dn=1-tanh**2=1-o**2
# n.grad=0.5

