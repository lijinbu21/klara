![[docs/images/Pasted image 20241114104225.png]]
## 初始化和网格加载   :   顶点矩阵和面片矩阵实现
## 计算微分算子
----
### 梯度
 `self.diff_ops.grad = poisson_sys_mat.igl_grad` ，稀疏矩阵 SparseMat
3F×V (F是面片数量，V是顶点数量)，计算每个面片局部变形，
调用igl库计算  ==`grad = igl.grad(vertices, faces)  # (3F, V)`==
grad = rearrange_igl_grad_elements(grad) 重排列 
原始格式: [x1...xn, y1...yn, z1...zn]
目标格式: [x1,y1,z1, x2,y2,z2, ...]
**每一行表示一个面片上一个方向的梯度**

---
### 质量矩阵
 mass = _get_mass_matrix(vertices, faces, is_sparse)  # (3F, 3F)
 areas = igl.doublearea(vertices, faces) *0.5  面积是double area的一半
  d_area = np.hstack((d_area, d_area, d_area))  设置x\y\z分量权重都相同，也相当于把面积数组重复三次，使这个面积对应三个分量
  构建稀疏对角矩阵
1. 面积矩阵初次生成是一个数组，一个面片有一个double对应
2. 扩展到三分量是一起扩展
原始：[2.0, 3.0, 1.5]
扩展后：[2.0, 3.0, 1.5, 2.0, 3.0, 1.5, 2.0, 3.0, 1.5]
        |----x----|  |----y----|  |----z----|
3. 因为是一个稀疏矩阵，转换为对角矩阵存储，最终的格式是

```
这是有三个面片的网格的情况，总之就是先重复一遍，再另外一个分量
[2.0 0   0   0   0   0   0   0   0  ] 第一个面片的x分量
[0   3.0 0   0   0   0   0   0   0  ]
[0   0   1.5 0   0   0   0   0   0  ] 第n个面片的x分量
[0   0   0   2.0 0   0   0   0   0  ] 第一个面片y分量
[0   0   0   0   3.0 0   0   0   0  ]
[0   0   0   0   0   1.5 0   0   0  ]
[0   0   0   0   0   0   2.0 0   0  ]
[0   0   0   0   0   0   0   3.0 0  ]
[0   0   0   0   0   0   0   0   1.5]
```
这个是为了与梯度算子维度匹配，G是3F×V
```
# 梯度算子G是9×4（假设顶点数4）的矩阵（3个面片 × 3个方向 × 4个顶点）：
G = [g11  g12  g13  g14]  # 第1个面片的x方向梯度
    [g21  g22  g23  g24]  # 第2个面片的x方向梯度
    [g31  g32  g33  g34]  # 第3个面片的x方向梯度
    [g41  g42  g43  g44]  # 第1个面片的y方向梯度
    [g51  g52  g53  g54]  # 第2个面片的y方向梯度
    [g61  g62  g63  g64]  # 第3个面片的y方向梯度
    [g71  g72  g73  g74]  # 第1个面片的z方向梯度
    [g81  g82  g83  g84]  # 第2个面片的z方向梯度
    [g91  g92  g93  g94]  # 第3个面片的z方向梯度
```
gij表示第i个梯度分量相对于第j个顶点的偏导数，每个面片只与三个顶点相关，对于三角网格，即使V变大了，每一行也只有三个非零元素，这同样可以表示为一个稀疏矩阵

----
### 拉普拉斯矩阵
WKS： 基于Laplace算子的特征值分解，描述网格局部几何特征的方法，关注的是频率域的特征，描述每个点再不同频率下的响应，低频——全局特征，高频——局部细节。这个特征的优点：
 旋转不变性、 等距不变性、对噪声稳定、捕捉多尺度特征。
 用这个特征是为了做形状匹配和特征对应，使用它计算原网格和目标网格之间的对应关系，进行特征点匹配，利用特征相似度来调整约束的权重。

```
wks计算流程
1. 预处理
   ├── 检查网格连通性
   └── 构建拉普拉斯和质量矩阵
   # 构建cotangent拉普拉斯矩阵
	L = -igl.cotmatrix(self.vertices, self.faces)

# 构建Voronoi质量矩阵
	M = igl.massmatrix(self.vertices, self.faces, igl.MASSMATRIX_TYPE_VORONOI)```

2. 特征值分解Lφ = λMφ
   ├── 求解广义特征值问题
   ├── 获取特征值和特征向量
   └── 归一化特征向量

3. WKS计算
   ├── 计算能量参数(delta, sigma)
   ├── 生成能量采样点(es)
   ├── 计算高斯权重(coef)
   └── 计算最终WKS特征

4. 特征使用
   ├── 形状匹配
   ├── 特征点检测
   └── 对应关系建立

# 1. 特征点检测
def detect_features(wks):
    """使用WKS检测显著点"""
    return find_peaks(wks.max(axis=1))

# 2. 形状匹配
def match_shapes(wks1, wks2):
    """使用WKS进行形状匹配"""
    return compute_similarity(wks1, wks2)

# 3. 变形约束
def get_constraints(source_wks, target_wks):
    """基于WKS特征确定变形约束点"""
    return find_corresponding_points(source_wks, target_wks)
```
WKS的计算过程：
- 特征值分解:
- 求解方程 Lφ = λMφ
- λ是特征值（频率）
- φ是特征向量（振动模式）

---
### 泊松系统构建
buildIndicatorMatrix(): 构建约束指示矩阵

Ka的构建
# src/geometry/PoissonSystem.py

```
def _build_indicator_matrix(self):
    """构建约束指示矩阵"""
    num_vertices = self.num_vertices
    num_handles = len(self.handle_indices)
    
    # 构建稀疏矩阵
    indices = torch.zeros((2, num_handles), dtype=torch.long)
    values = torch.ones(num_handles)
    
    # 设置约束点的位置
    indices[0, :] = torch.arange(num_handles)
    indices[1, :] = torch.tensor(self.handle_indices)
    
    # 创建指示矩阵
    indicator = SparseMat(
        indices, values,
        torch.Size([num_handles, num_vertices])
    )
    
    # 计算 K = indicator^T @ indicator
    self.indicator_product = indicator.t() @ indicator
```

```
# 1. L^T L 项
mat_to_fac = mat_to_fac.transpose() @ mat_to_fac  # 确保正定性

# 2. λK 项
mat_to_fac = mat_to_fac + self.constraint_lambda * indicator_product
```

这个函数返回了左端项
 def _compute_matrix_to_factorize(self):




### 雅可比矩阵
    J = self.grad.multiply_with_dense(vertices)
   重排j，雅可比从[num_face×3×3]重排为[num_face9]的展平形式
   # 重排为交错形式 [x1,y1,z1, x2,y2,z2, ...]

    output = torch.zeros_like(input).reshape(-1)

    output[::3] = Tx.reshape(-1)

    output[1::3] = Ty.reshape(-1)

    output[2::3] = Tz.reshape(-1)



---

代码实现完整

最后的求解是


mat_to_fac = self._compute_matrix_to_factorize()
   # 使用SPLU进行LU分解
   self.L_fac = SPLUSolver(mat_to_fac)  # 存储分解结果
   rhs = grad.transpose() @ mass  # 梯度转置与质量矩阵相乘
   rhs = self.rhs.multiply_with_dense(jacobians_)
        rhs = self.L.transpose().multiply_with_dense(rhs)
        rhs = rhs + self.constraint_lambda * (
            self.indicator_matrix.transpose().multiply_with_dense(constraint)
        )
	  

        # solve the constrained least squares
        v_sol: Shaped[Tensor, "num_vertex 3"] = SPLUSolveLayer.apply(
            self.L_fac,
            rhs[None, ...],
        )[0, ...]
        v_sol = v_sol.type_as(self.J)
        return v_sol
       形状：[num_vertex × 3]


---
梳理了泊松系统的全部流程，以及重构了代码结构，下一步先把所有的矩阵调整好，重新求解；第二个是优化过程中，源代码用的torch，由于C++没有自动微分，自己的实现代码不能保证正确，对于这个最小二乘问题，找了一个ceres优化库用来实现自动微分

---
之前忽略的点

我的泊松求解不稳定，发现源代码有一个引入四元数场表示局部选咋混编写，用来防止相邻面片的差异过大，这个部分要写进去

预处理阶段使用wks来建立对应关系