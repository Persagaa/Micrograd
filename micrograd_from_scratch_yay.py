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
    def __init__(self,data,_children=(),_op=''):
        self.data=data
        self._prev =set(_children)
        self._op=_op
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
a=Value(2.0)
print(a)
b=Value(-3.0)
# a.data+b.data和a.__add__(b)的区别在于前者是直接访问对象的属性进行加法运算，而后者是调用对象的__add__方法来实现加法运算。
# 在Python中，运算符重载允许我们通过定义特殊方法（如__add__）来改变运算符的行为，使其适用于自定义对象。
# 当我们使用a+b时，实际上会调用a对象的__add__方法，并将b作为参数传递给该方法，从而实现自定义的加法运算逻辑。
c=Value(10.0)
# print(a*b+c)
d=a*b+c
# d._prev是一个集合，包含了与d相关的子对象。在这个例子中，d是通过a、b和c的运算得到的，因此d._prev包含了a、b和c这三个Value对象。
print(d._prev)
print(d._op)

