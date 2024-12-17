1.  G.outerSize()：

- 对于列优先存储（ColMajor，Eigen默认），返回列数

- 对于行优先存储（RowMajor），返回行数

- 用于遍历稀疏矩阵的非零元素

2. SparseMat::InnerIterator：

- 用于遍历稀疏矩阵中非零元素的迭代器

- 对于列优先存储，它按列遍历每一列中的非零元素

- 对于行优先存储，它按行遍历每一行中的非零元素


```
 #include <Eigen/Sparse>

void explainSparseMatrix() {
    typedef Eigen::SparseMatrix<double> SparseMat;
    
    // 创建一个 5x5 的稀疏矩阵
    SparseMat mat(5, 5);
    
    // 使用 triplets 填充矩阵
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.push_back(Eigen::Triplet<double>(0, 0, 1.0));  // (row, col, value)
    triplets.push_back(Eigen::Triplet<double>(1, 1, 2.0));
    triplets.push_back(Eigen::Triplet<double>(0, 4, 3.0));
    mat.setFromTriplets(triplets.begin(), triplets.end());
    
    // 遍历方式1：按列遍历（默认列优先存储）
    std::cout << "按列遍历：\n";
    for (int k = 0; k < mat.outerSize(); ++k) {  // 遍历每一列
        std::cout << "列 " << k << " 的非零元素：\n";
        for (SparseMat::InnerIterator it(mat, k); it; ++it) {
            std::cout << "行: " << it.row()        // 行索引
                      << ", 列: " << it.col()      // 列索引
                      << ", 值: " << it.value()    // 元素值
                      << std::endl;
            
            // 也可以修改值
            it.valueRef() = 1.5;  // 修改当前非零元素的值
        }
    }
    
    // 遍历方式2：使用 coeffRef 直接访问元素
    double val = mat.coeffRef(0, 0);  // 获取元素值
    mat.coeffRef(0, 0) = 2.0;         // 修改元素值
    
    // 插入新的非零元素
    mat.insert(2, 3) = 4.0;  // 在 (2,3) 位置插入值 4.0
    
    // 压缩矩阵以优化存储
    mat.makeCompressed();
    
    // 获取矩阵信息
    std::cout << "非零元素个数: " << mat.nonZeros() << std::endl;
    std::cout << "行数: " << mat.rows() << std::endl;
    std::cout << "列数: " << mat.cols() << std::endl;
}
```
![[docs/images/Pasted image 20241115174101.png]]