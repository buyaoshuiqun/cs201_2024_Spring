

一、dp：

看了题解才明白是这么写：

![联想截图_20240312113420](D:\cjh\大学\作业＆考试\大一下\数算\Homework\Week3\联想截图_20240312113420.png)

PS: 还应该补上一句判断dp[i]+1>maxlen，保证在子列中找到最大拦截导弹数目

有了状态转移方程之后遍历每个状态即可——for j in range(n)	for i in range(j)

二、快排和归并排序

```python
def quicksort(arr, left, right):
    if left < right:
        partition_pos = partition(arr, left, right)
        quicksort(arr, left, partition_pos - 1)
        quicksort(arr, partition_pos + 1, right)


def partition(arr, left, right):
    i = left + 1
    j = right
    pivot = arr[left]
    while i <= j:
        while i <= right and arr[i] <= pivot:
            i += 1
        while j >= left and arr[j] > pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    if arr[j] < pivot:
        arr[j], arr[left] = arr[left], arr[j]
    return j


arr = [22, 11, 88, 66, 55, 77, 33, 44]
quicksort(arr, 0, len(arr) - 1)
print(arr)
```

```python
#global ans在oj的编译器里面居然会CE，然后自己有点想法——如果想在merge里面加上一个ans
#然后在外面叠加，那么可能需要return两个值———于是自己就不会做，看了答案找到解法了
#最后还忘记把len==1的情况的return改成return ls,0了

#mergeSort是求逆序数的经典算法，求tao的时候需要在merge合并函数里面，把right数组
#里面的数需要交换的次数加在计数器上，而left里面不要动！！！
#递归只看n-1的结果和现在的过程————n-1的结果是left已经排好，现在merge里
#while里面的过程就相当于把right里面的数换到left里面，所以不要动left！！！
#right需要交换的次数==现在的位置-应该在的位置（相减所以用列表序号可以代替二者）==len(left)+r - len(result)

def mergeSort(ls):
    if len(ls) <= 1:
        return ls, 0
    mid = int(len(ls)/2)
    left, ans_left = mergeSort(ls[0:mid])
    right, ans_right = mergeSort(ls[mid:])
    mergedlist, ans_merge = merge(left,right)
    return mergedlist, ans_left + ans_right + ans_merge

def merge(left, right):
    ans = 0
    r, l=0, 0
    result = []
    while l<len(left) and r<len(right):
        if left[l] <= right[r]:
            result.append(left[l])
            l+=1
        else:
            result.append(right[r])
            r+=1
            ans += len(left)+r-len(result)
    result += list(left[l:])
    result += right[r:]
    return result, ans

while True:
    n = int(input())
    if n == 0:
        break
    else:
        ls = []
        for i in range(n):
            ls.append(int(input()))
        _, ans = mergeSort(ls)
        print(ans)
```



三、bfs和dfs

```python
# 
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
    
def prase_tree(arr):
    stack = []
    for token in arr:
        node = TreeNode(token)
        if token.isupper():
            if not stack:
                stack.append(node)
            else:
                b = stack.pop()
                a = stack.pop()
                node.left, node.right = a, b
                stack.append(node)
        else:
            stack.append(node)
    return node

def bfs(root):
    if not root:
        return []
    else:
        ans, queue = [], [root]
        while queue:
            node = queue.pop(0)
            ans.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return ans


n = int(input())
for _ in range(n):
    arr = input()
    root = prase_tree(arr)
    ans = bfs(root)
    ans.reverse()
    print(''.join(ans))
```



四、手搓堆

思路：半抄半大致理解了一下BinHeap的实现方式，主要是要理解几个辅助函数

首先理解class堆的建立方式（和heapq包里面的不一样，heapq里面无法获取下标对应的数）：

①用数组存储val
②root和left、right在HeapList里面有下标关系：i,i×2,i×2+1（这个*太智能了）[初始列表放一个0的原因]
③val有root比children小

之后是percUp函数：由于不清楚添加位置，所以只好把insert的数k加到List的末尾（append），此时就需要调整k的位置以满足条件③。percUp接受一个参数i即k的index，不断和i//2即k的parent比较大小和交换直到i//2 == 0（注意i = i // 2要写在if外面不然会RE）

之后是percDown函数：这个稍微复杂一些。delMin函数删除的就是堆顶的元素，删除之后谁来当新的堆顶元素呢？如果直接把堆顶最小的儿子拿过来，那么得递归不断调儿子的儿子的儿子......似乎时间复杂度会大很多。而因为最后一个元素一定是叶节点，直接拿过来也不会改变前面的顺序，那么把它往下移到合适的位置就好了。所以不断找子节点直到i*2>currentSize即没有子节点，每一步把这个元素和最小的儿子比较和交换

最小儿子用minChild，要注意的是minChild只用在了percDown里面，而percDown判断了i*2<currentSize为True之后才会调用minChild，因此minChild得到的参数一定有儿子，但是可能没有右儿子（i×2+1>Size)需要判断一下。如果有右儿子才能正常比较左右谁大

代码

```python
# 
#50min
class BinHeap:
    def __init__(self):
        self.HeapList = [0]
        self.currentSize = 0
    
    def percUp(self,i):
        while i // 2 > 0:
            if self.HeapList[i] < self.HeapList[i//2]:
                temp = self.HeapList[i//2]
                self.HeapList[i//2] = self.HeapList[i]
                self.HeapList[i] = temp
            i = i // 2

    def insert(self,k):
        self.HeapList.append(k)
        self.currentSize += 1
        self.percUp(self.currentSize)
    
    def minChild(self, i):      #只用在percDown里面，而percDown中判断i*2<size意思是取
        if i*2 + 1 > self.currentSize:      #非叶子节点，因此一定有儿子，但是可能没右儿子
            return i*2
        else:
            if self.HeapList[i*2] < self.HeapList[i*2 + 1]:
                return i*2
            else:
                return i*2 + 1
    
    def percDown(self, i):
        while i*2 <= self.currentSize:
            mc = self.minChild(i)
            if self.HeapList[i] > self.HeapList[mc]:    #一不小心写反了，但是测试用例过了
                temp = self.HeapList[mc]    #因为没有反复delMin,否则会发现排序反了
                self.HeapList[mc] = self.HeapList[i]
                self.HeapList[i] = temp
            i = mc
    
    def delMin(self):
        retval = self.HeapList[1]
        self.HeapList[1] = self.HeapList[self.currentSize]
        self.HeapList.pop()
        self.currentSize -= 1
        self.percDown(1)
        return retval
    
    def buildHeap(self, ls):
        i = len(ls) // 2
        self.currentSize = len(ls)
        self.HeapList = [0] + ls
        while i > 0:
            self.percDown(i)
            i = i - 1

n = int(input())
heap = BinHeap()    #一不小心把这个写在循环里面了
for _ in range(n):
    arr = input().split()
    if arr == ['2']:
        print(heap.delMin())
    else:
        u = int(arr[-1])
        heap.insert(u)
```

五、手搓avl

思路：对着题解搬运之后逐步理解，难点在两个rotate和树型的判断。

左旋要领是让z.left即y作为root——只需让z成为y.left，在此之前避免丢失原本的y.left要用T2存储；交换之后z.right可以空出，就让z.right = T2<img src="D:\cjh\大学\作业＆考试\大一下\数算\Homework\Week6\Weixin Image_20240331191606.jpg" alt="Weixin Image_20240331191606" style="zoom:25%;" />
右旋要领是让z.right即x作为root——只需让z成为x.right，在此之前存储T1 = x.right；交换之后z.left = T1
<img src="D:\cjh\大学\作业＆考试\大一下\数算\Homework\Week6\Weixin Image_20240331191601.jpg" alt="Weixin Image_20240331191601" style="zoom:25%;" />

再来说说_insert方法，也是我个人认为最麻烦和困难的一个方法：

​	实际上从参数和返回值里面两个node来看，_insert方法实际目的是 **排序/调整node和下面两个子树**。包含两个部分——

①加入val，不断利用二叉搜索性递归，not node的时候建立并返回一个Tree Node(val)。然后height+=1
②处理balance：得到balance之后，由于node是一步步加上去的，因此balance最坏只会有±2。那么就只用旋转1次（或2次，下面会说）让树balanced。以LL和LR为例，LL(和RR)是最简单的——只需要右旋一次就好了；LR复杂一些，但是可以转化为LL——把root.left左旋一次。

如何判断第一个L/R和第二个L/R：
①balance是左子树高--右子树高，balance>1是L/＜1是R
②前面说了balance最坏±2，即最多左比右高2(以L形为例)。而且加入val之前是平衡的，因此val在哪哪里就不平衡——二叉搜索性val<root.left.val就在左，为LL；反之为LR。

代码

```python
# 
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1

class AVL:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
        else:
            self.root = self._insert(val, self.root)
    
    def _insert(self, val, node):
        if not node:
            return TreeNode(val)
        elif val < node.val:
            node.left = self._insert(val, node.left)
        else:
            node.right = self._insert(val, node.right)
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        
        balance = self._get_balance(node)

        if balance > 1:
            if val < node.left.val:     #这种情况新的val会加到node.left的左子树
                return self._rotate_right(node) #因此node.left应该是L型（之前的树是平衡的）
            else:   #这种情况val加到node.left的右子树，那么node.left应该是R型（右高）
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)
        
        if balance < -1:
            if val > node.right.val:
                return self._rotate_left(node)
            else:
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)
        
        return node

    def _get_height(self, node):
        if not node:
            return 0
        return node.height
    
    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)
    
    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        #没办法直接根据旋转的情况算出来，可能没有z.right
        #而前面存储的left和right的高度没变，可以用
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y
    
    def _rotate_right(self, z):
        x = z.left
        T1 = x.right
        x.right = z
        z.left = T1
        z.height = 1 + max(self._get_height(z.left), self._get_height(###z###.right))
        ##找了半天bug才发现自己把上面那个z写成x了
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x
    
    def preorder(self, node):
        if not node:
            return []
        return [str(node.val)] + self.preorder(node.left) + self.preorder(node.right)

n = int(input())
nums = list(map(int,input().strip().split()))

avl = AVL()
for num in nums:
    avl.insert(num)
print(' '.join(avl.preorder(avl.root)))
```

六、并查集

思路：先学习了一下并查集，找到csdn上面的一个大大很通俗的解释（[spm=1018.2226.3001.4187](https://blog.csdn.net/the_ZED/article/details/105126583?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171186743516800182152951%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171186743516800182152951&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-105126583-null-null.142^v100^pc_search_result_base8&utm_term=%E5%B9%B6%E6%9F%A5%E9%9B%86&spm=1018.2226.3001.4187))。明确并查集是用parent列表存储每个index对应的parent之后，只需要建立一个0到n的列表（保持index对应），再把条件对应的同学合并即可
最后输出上限：**有多少parent就有多少组织**，那么只用遍历一遍找到parent[x] == x有多少即可

find方法是递归找parent，终止条件即parent是自己本身（没有被并入别的组织——被并入之后就认了别人当parent了！）

union方法：参数两个同学（一般是x,y -> int），找到两个同学的parent并比较，如果一样就不用union；如果不一样，在__init里面我们定义了属性rank（一个用来存储某种地位高度的列表），谁地位高谁是parent（修改parent[xset] = yset）

简化版本就直接随便把谁当大哥就行（）

七、约瑟夫问题

```python
from collections import deque
# 先使⽤pop从列表中取出，如果不符合要求再append回列表，相当于构成了⼀个圈
def hot_potato(name_list, num):
	queue = deque()
	for name in name_list:
		queue.append(name)
	while len(queue) > 1:
		for i in range(num):
			queue.append(queue.popleft()) # O(1)
		queue.popleft()
	return queue.popleft()

while True:
	n, m = map(int, input().split())
	if {n,m} == {0}:
		break
	monkey = [i for i in range(1, n+1)]
	print(hot_potato(monkey, m-1))
```

