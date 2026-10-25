class Value:
    """stores a single scalar value and its gradient"""
    """存储一个单独的标量值及其梯度"""
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0                   
        #internal variables used for autograd graph construction
        #用于自动梯度图构建的内部变量
        self._backward = lambda: None
        #为什么这些变量前面有_？
        #_表示这些变量是私有的，不应该被外部访问
        self._prev = set(_children)
        self._op = _op
        #the op that produced this node,for graphviz / debugging /etc
        #生成此节点的操作，用于graphviz/调试等
        #graphviz是一个用于可视化图形的工具

    def __add__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        #如果other不是Value类型，则将其转换为Value类型
        #isinstance()函数用于判断一个对象是否是一个类的实例
        #isinstance(object,classinfo)object:要判断的对象\classinfo:要判断的类
        out=Value(self.data+other.data,(self,other),'+')
        #为什么要单独列一个方法做backward?而不是直接在__add__方法里写？
        #因为这样可以使代码更加清晰，将backward逻辑与__add__逻辑分离
        def _backward():
            self.grad+=out.grad
            other.grad+=out.grad
        out._backward=_backward
        #为什么这里要加一个out._backward=_backward?
        #因为这样可以使out对象拥有一个_backward方法，这个方法可以在out对象被反向传播时调用
        return out
    def __mul__(self,other):
        other=other if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad+=other.data*out.grad
            other.grad+=self.data*out.grad
        out._backward=_backward
        return out

    def __pow__(self,other):
        assert isinstance(other,(int,float)),"only supporting int/float powers for now"#暂时只支持int/float幂
        #assert语句用于检查一个条件是否为真，如果为假，则抛出一个AssertionError异常
        out=Value(self.data**other,(self,),f'**{other}')
        def _backward():
            self.grad+=(other*self.data**(other-1))*out.grad
        out._backward=_backward
        return out

    def relu(self):
        #ReLU函数
        #ReLU函数是深度学习中常用的激活函数之一，
        # 它的定义是：如果输入大于0，则输出为输入，否则输出为0
        #ReLU函数的导数是：如果输入大于0，则输出为1，否则输出为0
        out=Value(0 if self.data<0 else self.data,(self,),'ReLU')
        def _backward():
            self.grad+=(out.data>0)*out.grad
        out._backward=_backward
        return out

    def backward(self):
        #反向传播
        #反向传播是深度学习中常用的算法之一，
        # 它的定义是：从后往前计算梯度
        #反向传播的算法是：从后往前计算梯度，直到到达输入层
        #topological order all of the children in the graph
        #对图中的所有子节点进行拓扑排序
        topo = []                    # ① 存放排好序的节点
        visited = set()              # ② 防止重复访问 / 死循环
        def build_topo(v):           # ③ 内部递归函数
            if v not in visited:     # ④ 已访问过就直接返回
                visited.add(v)       # ⑤ 打标记
                for child in v._prev:      # ⑥ 先递归所有输入
                    build_topo(child)
                topo.append(v)       # ⑦ 后把自己加入列表 ★关键
        build_topo(self)             # ⑧ 从 loss 出发点火

    def backward(self):
        topo = []                    # ① 存放排好序的节点
        visited = set()              # ② 防止重复访问 / 死循环
        def build_topo(v):           # ③ 内部递归函数
            if v not in visited:     # ④ 已访问过就直接返回
                visited.add(v)       # ⑤ 打标记
                for child in v._prev:      # ⑥ 先递归所有输入
                    build_topo(child)
                topo.append(v)       # ⑦ 后把自己加入列表 ★关键
        build_topo(self)             # ⑧ 从 loss 出发点火

        #go one variable at a time and apply the chain rule to get its gradient
        #一个一个地遍历变量并应用链式法则来获取其梯度
        self.grad=1
        #为什么这里要设为1？
        #因为反向传播的算法是：从后往前计算梯度，直到到达输入层
        #所以，在计算第一个节点的梯度时，需要将其设为1
        #为什么要reversed?
        #因为反向传播的算法是：从后往前计算梯度，直到到达输入层
        for v in reversed(topo):
            v._backward()#反向传播
            #为什么这里要加一个v._backward()?
            #因为这样可以使v对象拥有一个_backward方法，这个方法可以在v对象被反向传播时调用
    
    #为什么这里不需要设置梯度的变换了？
    #因为这里的梯度变换已经在__add__、__mul__、__pow__等方法中设置过了

    def __neg__(self):#-self
        return self*-1#调用__mul__方法，所以不用重新说明梯度的变化
    
    def __radd__(self,other):#other+self
        return self+other#调用__add__方法，所以不用重新说明梯度的变换
        #他和__add__方法的区别是，__add__方法是在self对象前面加，而__radd__方法是在other对象前面加

    def __sub__(self,other):#self-other
        return self+(-other)#调用__add__方法，所以不用重新说明梯度的变换
        #他和__add__方法的区别是，__add__方法是在self对象前面加，而__sub__方法是在other对象前面减

    def __rsub__(self,other): #other-self
        return other+(-self)

    def __rmul__(self,other):#other*self
        return self*other

    def __truediv__(self,other):#self/other
        return self*other**-1

    def __rtruediv__(self,other):#other/self
        return other*self**-1

    def __repr__(self):
        return f"Value(data={self.data},grad={self.grad})"

#micrograd的代码中，Value类是核心类，它实现了反向传播算法，并且提供了许多操作符重载方法，使得我们可以像操作普通数值一样操作Value对象。
#Value类中的每个方法都实现了反向传播算法的一部分，例如__add__、__mul__、__pow__等方法实现了加法、乘法和幂运算的反向传播算法，而backward方法则实现了整个反向传播算法。
#micrograd和pyTorch里的loss.backward()有什么区别？
#micrograd的backward方法是在Value类中实现的，而pyTorch的backward方法是在Tensor类中实现的。
#关系是什么？#Value类是Tensor类的父类，Tensor类是pyTorch中的核心类，它实现了许多操作符重载方法，使得我们可以像操作普通数值一样操作Tensor对象。

#Value类的骨架是：data的作用：存储数值，
#                grad的作用：存储梯度，
#               _prev的作用：存储前向传播过程中的所有节点，
#               _backward的作用：存储反向传播过程中的所有节点，
#               backward的作用：实现反向传播算法，
#               __add__、__mul__、__pow__等方法的作用：实现加法、乘法和幂运算的反向传播算法，
#               __repr__的作用：返回Value对象的字符串表示，
#               _children的作用：返回Value对象的所有子节点，
#               _op的作用：返回Value对象的操作符。

# tanh的局部导数：tanh(x)的导数是1-tanh(x)**2

# 为什么backward()需要先拓扑排序，再逆序执行？
# 因为反向传播的算法是：从后往前计算梯度，直到到达输入层
# 所以，在计算第一个节点的梯度时，需要将其设为1
# 而拓扑排序的作用是：将图中的所有节点按照拓扑顺序排列，这样就可以按照拓扑顺序计算梯度了
# 逆序执行的作用是：从后往前计算梯度，直到到达输入层
# 拓扑排序是什么？
# 拓扑排序是一种对有向无环图（DAG）进行排序的方法，使得对于每一条有向边(u,v)，顶点u在排序中都出现在v之前。
# 从u指向v,所以先从u开始拓扑排序，还是v？
# 从u开始拓扑排序，因为u是v的前驱节点，所以u应该先于v被访问。

