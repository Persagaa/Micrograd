import math
#math是数学库，提供了数学函数和常量
import numpy as np
#numpy是一个科学计算库，提供了高效的数组操作和数学函数
import matplotlib.pyplot as plt
#%matplotlib inline是一个魔法命令，用于在Jupyter Notebook中启用交互式绘图。
# matplotlib是一个绘图库，提供了丰富的绘图功能，可以生成各种类型的图表和可视化效果。
def f(x):
    return 3*x**2-4*x+5

print(f(3.0))

# 绘制函数图像
xs=np.arange(-5,5,0.25)
ys=f(xs)
print(ys)
plt.plot(xs,ys)
# plt.show()

h =0.001
x=3.0
# 计算导数的近似值
print((f(x+h)-f(x))/h)
#如果h足够小，这个近似值将非常接近函数在x=3.0处的真实导数值。
#但如果h过小，可能会引入数值误差，因为浮点数运算的精度限制，因此需要选择一个合适的h值来平衡精度和稳定性。
x=2/3
print((f(x+h)-f(x))/h)

a=2.0
b=-3.0
c=10.0
d=a*b+c
print(d)

#判断a会对d的值产生多大的影响，计算d对a的导数
h=0.0001
a=2.0
b=-3.0
c=10.0
d1=a*b+c
a+=h
d2=a*b+c
print('d1',d1)
print('d2',d2)
print('slope', (d2-d1)/h)

#判断b会对d的值产生多大的影响，计算d对b的导数
h=0.0001
a=2.0
b=-3.0
c=10.0
d1=a*b+c
b+=h
d2=a*b+c
print('d1',d1)
print('d2',d2)
print('slope', (d2-d1)/h)





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
a=Value(2.0,label="a")
print(a)
b=Value(-3.0,label="b")
# a.data+b.data和a.__add__(b)的区别在于前者是直接访问对象的属性进行加法运算，而后者是调用对象的__add__方法来实现加法运算。
# 在Python中，运算符重载允许我们通过定义特殊方法（如__add__）来改变运算符的行为，使其适用于自定义对象。
# 当我们使用a+b时，实际上会调用a对象的__add__方法，并将b作为参数传递给该方法，从而实现自定义的加法运算逻辑。
c=Value(10.0,label="c")
#python中分号的作用是将多条语句写在同一行上，分号用于分隔不同的语句，使它们在同一行中被执行。
e=a+b;e.label="e"
# print(a*b+c)
d=e+c;d.label="d"
f=Value(-2.0,label="f")
L=d*f;L.label="L"
# d._prev是一个集合，包含了与d相关的子对象。在这个例子中，d是通过a、b和c的运算得到的，因此d._prev包含了a、b和c这三个Value对象。
print(d._prev)
print(d._op)

from numpy import arange
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
        dot.node(name=uid,label="{%s | data %.4f | grad %.4f}"%(n.label, n.data, n.grad),shape="record")
        if n._op:
            #如果节点n有操作类型（_op不为空），则为该操作创建一个圆形节点，并将其与节点n连接起来。
            dot.node(name=uid+n._op,label=n._op)
            #并且将操作节点与节点n连接起来，形成有向边。
            dot.edge(uid+n._op,uid)

    for n1,n2 in edges:
        #连接所有的边，将节点n1和节点n2连接起来，形成有向边。
        dot.edge(str(id(n1)),str(id(n2))+n2._op)

    return dot
print(draw_dot(L))


#L=d*f
#dL/dd=? f
#(f(x+h)-f(x))/h
#dL/df=(d(f+h)-d*f)/h
#(d*f+d*h-d*f)/h


#链式法则
#dd/dc=1.0
#dd/de=1.0
# d=c+e
# WANT:dL/dc=dL/dd*dd/dc

def lol():

    h=0.001
    a=Value(2.0,label="a")
    b=Value(-3.0,label="b")
    c=Value(10.0,label="c")
    e=a+b;e.label="e"
    d=e+c;d.label="d"
    f=Value(-2.0,label="f")
    L=d*f;L.label="L"
    L1=L.data

    a=Value(2.0,label="a")
    b=Value(-3.0,label="b")
    c=Value(10.0,label="c")
    e=a+b;e.label="e"
    d=e+c;d.label="d"
    f=Value(-2.0,label="f")
    L=d*f;L.label="L"
    L2=L.data+h
    return ((L2-L1)/h)

print(lol())


# 微分
# 微分的定义：
# 微分的定义是：在函数f(x)中，如果存在一个数x0，使得函数f(x)在x0处的导数不为0，那么我们就说函数f(x)在x0处可微。
# 微分的意义是：函数f(x)在x0处的导数，就是函数f(x)在x0处的变化率。
# 微分的计算方法：
# 微分的计算方法是：在x0处，取一个很小的数h，然后计算函数f(x)在x0+h处的值，再计算函数f(x)在x0处的值，然后计算出函数f(x)在x0处的导数，即f(x)在x0处的变化率。
# 微分的计算公式：
# 微分的计算公式是：f(x)在x0处的导数 = (f(x0+h) - f(x0))/h
#控制梯度变化
#a.data+=0.01*a.grad
#b.data+=0.01*b.grad
#c.data+=0.01*c.grad
#f.data+=0.01*f.grad    

plt.plot(np.arange(-5,5,0.2),np.tanh(np.arange(-5,5,0.2)));plt.grid();
plt.show()