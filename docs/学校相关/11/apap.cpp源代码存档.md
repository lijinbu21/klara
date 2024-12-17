
```
//class Optimizer {
//public:
//	double learning_rate;
//
//	Optimizer(double lr) : learning_rate(lr) {}
//
//	// 使用数值差分计算梯度
//	Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& J, const double loss, const Eigen::SparseMatrix<double>& Kh,
//		const Eigen::SparseMatrix<double>& grad, const Eigen::SparseMatrix<double>& A,
//		const Eigen::SparseMatrix<double>& L, const Eigen::VectorXd& Th,
//		Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>& solver) {
//		Eigen::MatrixXd grad_J = Eigen::MatrixXd::Zero(J.rows(), J.cols());
//		double epsilon = 1e-6;
//		for (int i = 0; i < J.rows(); ++i) {
//			for (int j = 0; j < J.cols(); ++j) {
//				Eigen::MatrixXd J_plus = J;
//				J_plus(i, j) += epsilon;
//				// 使用 J_plus 计算 V_star 和新的 loss
//				Eigen::MatrixXd V_star = solver.solve(L.transpose() * (grad.transpose() * A * J_plus));
//				double loss_plus = (Kh * V_star - Th).squaredNorm();
//				grad_J(i, j) = (loss_plus - loss) / epsilon;  // 数值差分计算梯度
//			}
//		}
//		return grad_J;
//	}
//
//	// 梯度下降法更新参数
//	void step(Eigen::MatrixXd& J, const Eigen::MatrixXd& grad) {
//		J -= learning_rate * grad;  // 更新 J 参数
//	}
//};

bool APAP::Exec() {
	// 1. 初步求解
	testDebug();
	preparation();
	printf("1.1 runs \n");
	first_solve();
	printf("1.2 runs \n");

	// 2. 更新雅可比
	int max_iterations = 1;
	//Optimizer optimizer(0.01);
	double tolerance = 1e-6; // 收敛阈值

	Eigen::SparseMatrix<double> LTL = L.transpose() * L;
	Eigen::SparseMatrix<double> KaTKa = Ka.transpose() * Ka;
	Eigen::SparseMatrix<double> leftHandSide = LTL + lambda * KaTKa;
	for (int i = 0; i < leftHandSide.rows(); ++i) {
		leftHandSide.coeffRef(i, i) += 1e-6;  // 添加小的正则化项
	}

	solver.compute(leftHandSide);
	if (solver.info() != Eigen::Success) {
		std::cerr << "Decompo sition failed!" << std::endl;
		return false;
	}

	// 1. 检查矩阵维度是否匹配
	std::cout << "L dimensions: " << L.rows() << "x" << L.cols() << std::endl;
	std::cout << "grad dimensions: " << grad.rows() << "x" << grad.cols() << std::endl;
	std::cout << "A dimensions: " << A.rows() << "x" << A.cols() << std::endl;
	std::cout << "J dimensions: " << J.rows() << "x" << J.cols() << std::endl;
	std::cout << "Lambda value: " << lambda << std::endl;
	std::cout << "Matrix norm: " << leftHandSide.norm() << std::endl;



	for (int iter = 0; iter < max_iterations; ++iter) {
		printf("Iteration %d\n", iter);
		testDebug2(iter);
		// 3. 使用 first_solve 更新 V_star
		Eigen::SparseMatrix<double> J_sparse = J.sparseView();


		Eigen::MatrixXd rightHandSide = L.transpose() * (grad.transpose() * A * J_sparse);

		if (!rightHandSide.allFinite()) {
			std::cerr << "Invalid values in intermediate computation!" << std::endl;

			rightHandSide += lambda * Ka.transpose() * Ta;
			//rightHandSide += lambda * Kh.transpose() * Th;
			// 求解顶点位置 V*

			std::cout << "rightHandSide dimensions: " << rightHandSide.rows() << "x" << rightHandSide.cols() << std::endl;
			//Eigen::VectorXd rightHandSide_vec(rightHandSide.rows() * rightHandSide.cols());
			//for (int i = 0; i < rightHandSide.rows(); ++i) {
			//	for (int j = 0; j < rightHandSide.cols(); ++j) {
			//		rightHandSide_vec(i * 3 + j) = rightHandSide(i, j);
			//	}
			//}
			//std::cout << "convert to vec " << std::endl;

			//V_star = solver.solve(rightHandSide_vec);
			//std::cout << " solver.solve(rightHandSide_vec) " << std::endl;

			// 初始化结果矩阵
			V_star.resize(rightHandSide.rows() * 3); // 因为每个顶点有x,y,z三个坐标

			// 对每个坐标分量分别求解
			for (int i = 0; i < 3; ++i) {
				Eigen::VectorXd current_rhs = rightHandSide.col(i);



				// 检查右手边是否包含无效值
				if (!current_rhs.allFinite()) {
					std::cerr << "Warning: Right hand side contains invalid values for component " << i << std::endl;
					continue;
				}


				// 求解前检查数值范围
				std::cout << "RHS range for component " << i << ": "
					<< current_rhs.minCoeff() << " to " << current_rhs.maxCoeff() << std::endl;
				// 如果数值太大，进行归一化
				if (std::abs(current_rhs.maxCoeff()) > 1e6 || std::abs(current_rhs.minCoeff()) > 1e6) {
					double scale = std::max(std::abs(current_rhs.maxCoeff()), std::abs(current_rhs.minCoeff()));
					current_rhs /= scale;
				}

				Eigen::VectorXd partial_solution = solver.solve(current_rhs);

				// 检查解是否有效
				if (!partial_solution.allFinite()) {
					std::cerr << "Warning: Solution contains invalid values for component " << i << std::endl;
					continue;
				}


				// 限制解的范围
				for (int j = 0; j < rightHandSide.rows(); ++j) {
					double value = partial_solution(j);
					// 限制在合理范围内
					if (std::isnan(value) || std::isinf(value)) {
						value = 0.0;  // 或使用原始位置
					}
					else {
						value = std::max(std::min(value, 1e3), -1e3);  // 限制最大位移
					}
					V_star(j * 3 + i) = value;
				}
			}
			//V_star = solver.solve(rightHandSide);
			if (solver.info() != Eigen::Success) {
				std::cerr << "Solving failed!" << std::endl;
				return false;
			}
			std::cout << "V_star norm: " << V_star.norm() << std::endl;
			double max_val = V_star.maxCoeff();
			double min_val = V_star.minCoeff();
			std::cout << "V_star range: [" << min_val << ", " << max_val << "]" << std::endl;
			//std::cout << V_star.rows() << " \t" << V_star.cols() << std::endl;

			testPrint();


			//// 4. 计算损失
			//auto loss = (Kh * V_star - Th).squaredNorm();  // 计算损失
			//printf("runs here! iter : %d : 1 \n", iter);

			//// 5. 计算雅可比矩阵 J 的梯度
			//auto grad_J = optimizer.computeGradient(J, loss, Kh, grad, A, L, Th, solver);
			//printf("runs here! iter : %d : 2 \n", iter);

			//if (grad_J.norm() < tolerance) {
			//	break; // 提前终止
			//}

			//// 6. 使用梯度下降法更新雅可比矩阵 J
			//optimizer.step(J, grad_J);
			//printf("runs here! iter : %d : 3 \n", iter);

			// 7. 将新的顶点位置更新到网格
			for (int i = 0; i < mesh.vsize(); ++i) {
				if (3 * i + 2 >= V_star.size()) {
					std::cerr << "Index out of bounds at vertex " << i << std::endl;
					continue;
				}

				Eigen::Vector3d new_position = V_star.segment<3>(3 * i);

				// 添加有效性检查
				if (new_position.hasNaN() || !new_position.allFinite()) {
					std::cerr << "Invalid position for vertex " << i << ": "
						<< new_position.transpose() << std::endl;
					continue;
				}

				auto& v = mesh.get_vertex(igame::vHandle(i));
				v.x() = new_position(0);
				v.y() = new_position(1);
				v.z() = new_position(2);
			}
			printf("runs here! iter : %d : 4 \n", iter);
		}

		//mesh是目标移动位置，把它给resmesh
		for (size_t i = 0; i < resmesh.vsize(); i++)
		{
			auto& v = resmesh.get_vertex(igame::vHandle(i));
			auto& p = mesh.get_vertex(igame::vHandle(i));
			v.x() = p.x();
			v.y() = p.y();
			v.z() = p.z();
		}
		return true;
	}
}




void APAP::testDebug()
{
	using namespace std;
	//输出定位问题
	std::streambuf* original_cout_buffer = std::cout.rdbuf();

	// 打开文件并将 cout 重定向到该文件
	std::ofstream debug_file("debugAPAP_output.txt", std::ios::out);
	if (!debug_file.is_open()) {
		std::cerr << "Failed to open debug file." << std::endl;
		return;
	}
	std::cout.rdbuf(debug_file.rdbuf());
	//输出

	//1. 原始顶点位置和固定点、移动点坐标
	cout << "origin V " << endl;
	for (int i = 0; i < mesh.vsize(); ++i) {
		auto& p = mesh.get_pos(igame::vHandle(i));
		cout << "i :" << i << "\t" << p << endl;
	}
	cout << "fixed points" << endl;
	for (igame::vHandle vh : fixed) {
		cout << "before --- id :" << vh << "\t" << mesh.get_pos(igame::vHandle(vh)) << endl;
		cout << "after ---  id :" << vh << "\t" << resmesh.get_pos(igame::vHandle(vh)) << endl;
		cout << "-----------------------------------------------------" << endl;
	}
	cout << "moved points" << endl;
	for (igame::vHandle vh : moved) {
		cout << "before --- id :" << vh << "\t" << mesh.get_pos(igame::vHandle(vh)) << endl;
		cout << "after ---  id :" << vh << "\t" << resmesh.get_pos(igame::vHandle(vh)) << endl;
		cout << "-----------------------------------------------------" << endl;
	}
	std::cout.rdbuf(original_cout_buffer);
	debug_file.close();
}

void APAP::testDebug2(int i)
{
	//输出迭代过程中的顶点位置变化

	using namespace std;
	//输出定位问题
	std::streambuf* original_cout_buffer = std::cout.rdbuf();

	// 打开文件并将 cout 重定向到该文件
	std::ofstream debug_file("迭代debug.txt", std::ios::app);
	if (!debug_file.is_open()) {
		std::cerr << "Failed to open debug file." << std::endl;
		return;
	}
	std::cout.rdbuf(debug_file.rdbuf());
	cout << "idx   : \t" << i << endl;
	cout << "-------------------------------------------------------------------------------------------------------\n";
	cout << "-------------------------------------------------------------------------------------------------------\n";
	cout << "-------------------------------------------------------------------------------------------------------\n";
	cout << "-------------------------------------------------------------------------------------------------------\n";
	cout << "-------------------------------------------------------------------------------------------------------\n";
	cout << "-------------------------------------------------------------------------------------------------------\n";
	cout << "pos" << endl;

	for (int i = 0; i < mesh.vsize(); ++i) {
		auto& p = mesh.get_pos(igame::vHandle(i));
		cout << "i :" << i << "\t" << p << endl;
	}
	std::cout.rdbuf(original_cout_buffer);
	debug_file.close();
}

void APAP::testPrint()
{
	using namespace std;
	// 保存原始的 cout 缓冲区
	std::streambuf* original_cout_buffer = std::cout.rdbuf();

	// 打开文件并将 cout 重定向到该文件
	std::ofstream debug_file("printMatrix.txt");
	if (!debug_file.is_open()) {
		std::cerr << "Failed to open debug file." << std::endl;
		return;
	}
	std::cout.rdbuf(debug_file.rdbuf());

	cout << "origin V " << endl;
	for (int i = 0; i < mesh.vsize(); ++i) {
		auto& p = mesh.get_pos(igame::vHandle(i));
		cout << "i :" << i << "\t" << p << endl;
	}
	cout << "V_star diff  " << endl;
	for (int i = 0; i < 3 * resmesh.vsize() - 2; i += 3) {
		auto& p = mesh.get_pos(igame::vHandle(i / 3));
		cout << "i :" << i / 3 << "\t" << p << endl;
		cout << "i :" << i / 3 << "\t" << V_star(i) << "\t" << V_star(i + 1) << "\t" << V_star(i + 2) << endl;
		cout << " ------------------------------------------------------------------\n";
	}


	printSparseMatrix(L, "L");
	printSparseMatrix(A, "A");
	printSparseMatrix(Kh, "Kh");
	printSparseMatrix(Ka, "Ka");
	printSparseMatrix(grad, "grad");
	printDenseMatrix(J, "J");
	printVector(Th, "Th");
	printVector(Ta, "Ta");

	std::cout.rdbuf(original_cout_buffer);
	debug_file.close();

}

void APAP::first_solve() {
	// 1. 构造线性方程
	long long fsize = mesh.fsize(), vsize = mesh.vsize();
	J.resize(fsize * 3, 3);
	A.resize(3 * fsize, 3 * fsize);
	grad.resize(3 * fsize, vsize);
	pos_mesh_ref.resize(static_cast<int>(vsize));
	//cots.clear();//map结构不用clear，因为每次都会重新覆盖
	std::vector<Eigen::Triplet<double>> tripletlist_G;
	std::vector<Eigen::Triplet<double>> tripletlist_A;
	std::vector<Eigen::Triplet<double>> tripletlist_M;


	tripletlist_G.clear();
	tripletlist_A.clear();
	tripletlist_M.clear();


	// 计算三角形角的余切，填充余切权重
	for (auto& [h, f] : mesh.all_faces()) {
		auto vhs = f.get_vhs();
		auto& v0 = mesh.get_vertex(vhs[0]);
		auto& v1 = mesh.get_vertex(vhs[1]);
		auto& v2 = mesh.get_vertex(vhs[2]);

		Eigen::Vector3d p0(v0.x(), v0.y(), v0.z());
		Eigen::Vector3d p1(v1.x(), v1.y(), v1.z());
		Eigen::Vector3d p2(v2.x(), v2.y(), v2.z());

		Eigen::Vector3d v01 = p1 - p0;
		Eigen::Vector3d v02 = p2 - p0;
		Eigen::Vector3d v12 = p2 - p1;

		// 计算余切 cot(theta)
		double cos_theta_0 = v01.dot(v02) / (v01.norm() * v02.norm());
		double sin_theta_0 = (v01.cross(v02)).norm() / (v01.norm() * v02.norm());
		double cot_theta_0 = cos_theta_0 / sin_theta_0;

		double cos_theta_1 = (-v01).dot(v12) / (v01.norm() * v12.norm());
		double sin_theta_1 = ((-v01).cross(v12)).norm() / (v01.norm() * v12.norm());
		double cot_theta_1 = cos_theta_1 / sin_theta_1;

		double cos_theta_2 = (-v02).dot(-v12) / (v02.norm() * v12.norm());
		double sin_theta_2 = ((-v02).cross(-v12)).norm() / (v02.norm() * v12.norm());
		double cot_theta_2 = cos_theta_2 / sin_theta_2;

		// 根据余切值更新边的权重
		cots[std::make_pair(vhs[0], vhs[1])] = cot_theta_2;
		cots[std::make_pair(vhs[1], vhs[2])] = cot_theta_0;
		cots[std::make_pair(vhs[2], vhs[0])] = cot_theta_1;
	}
	//L矩阵更新
	Eigen::SparseMatrix<double> M;
	M.resize(static_cast<int>(vsize), static_cast<int>(vsize));
	for (const auto& [h, v] : mesh.all_vertices()) {
		double w = 0.0;
		if (vis[h]) {
			tripletlist_M.push_back(Eigen::Triplet<double>(static_cast<int>(h), static_cast<int>(h), 1.0));
			continue;
		}
		for (const auto& vh : mesh.adjoin_vertex(h)) {
			w += cots[std::make_pair(h, vh)] + cots[std::make_pair(vh, h)];
			tripletlist_M.push_back(Eigen::Triplet<double>(static_cast<int>(h), static_cast<int>(vh), -(cots[std::make_pair(h, vh)] + cots[std::make_pair(vh, h)]) / 2.0));
		}
		tripletlist_M.push_back(Eigen::Triplet<double>(static_cast<int>(h), static_cast<int>(h), 1.0));
	}
	M.setFromTriplets(tripletlist_M.begin(), tripletlist_M.end());
	L = M;


	for (int f = 0; f < fsize; ++f) {
		auto& face = mesh.get_face(igame::fHandle(f));
		auto vhs = face.get_vhs();
		auto& v0 = mesh.get_vertex(vhs[0]);
		auto& v1 = mesh.get_vertex(vhs[1]);
		auto& v2 = mesh.get_vertex(vhs[2]);

		Eigen::Vector3d p0(v0.x(), v0.y(), v0.z());
		Eigen::Vector3d p1(v1.x(), v1.y(), v1.z());
		Eigen::Vector3d p2(v2.x(), v2.y(), v2.z());

		Eigen::Matrix3d J_f;
		Eigen::Vector3d e1 = p1 - p0;
		Eigen::Vector3d e2 = p2 - p0;
		Eigen::Vector3d n(face.normal().x(), face.normal().y(), face.normal().z());
		J_f.col(0) = e1;
		J_f.col(1) = e2;
		J_f.col(2) = n;

		J.block<3, 3>(3 * f, 0) = J_f;

		// 1.3 计算每个三角形的面积 A
		double area = 0.5 * ((p1 - p0).cross(p2 - p0)).norm();
		for (int i = 0; i < 3; ++i) {
			tripletlist_A.emplace_back(3 * f + i, 3 * f + i, area);
		}

		A.setFromTriplets(tripletlist_A.begin(), tripletlist_A.end());

		//i,j,k是0，1，2个点
		Eigen::Vector3d deltaA = n.cross(p2 - p1) / 2 * area;
		Eigen::Vector3d deltaB = n.cross(p0 - p2) / 2 * area;
		Eigen::Vector3d deltaC = n.cross(p1 - p0) / 2 * area;

		// 1.4 填充梯度矩阵 G                                               
		for (int i = 0; i < 3; ++i) {
			tripletlist_G.emplace_back(3 * f + i, static_cast<int>(vhs[0]), deltaA(i)); //三元组三个参数： i， j， value  我们先看一下g怎么写  然后改这个i
			tripletlist_G.emplace_back(3 * f + i, static_cast<int>(vhs[1]), deltaB(i));
			tripletlist_G.emplace_back(3 * f + i, static_cast<int>(vhs[2]), deltaC(i));
		}
	}


	grad.setFromTriplets(tripletlist_G.begin(), tripletlist_G.end());




}

void APAP::preparation()
{

	// 准备与顶点位置无关的矩阵
	int vsize = mesh.vsize();

	// 构建锚点矩阵和目标位置 Ta
	std::vector<Eigen::Triplet<double>> tripletListKa;
	std::vector<Eigen::Triplet<double>> tripletListKh;
	//Kh.resize(1, vsize);
	Ka.resize(static_cast<int>(fixed.size()), vsize);   // 5 * 1058
	Kh.resize(static_cast<int>(moved.size()), vsize);   // 5 * 1058

	for (size_t i = 0; i < fixed.size(); ++i) {
		int vh_idx = static_cast<int>(fixed[i]);
		tripletListKa.emplace_back(static_cast<int>(i), vh_idx, 1.0);

	}
	for (size_t i = 0; i < moved.size(); ++i) {
		int vh_idx = static_cast<int>(moved[i]);
		tripletListKh.emplace_back(static_cast<int>(i), vh_idx, 1.0);
	}
	//Kh.insert(0, static_cast<int>(MovedPoint)) = 1.0;
	Ka.setFromTriplets(tripletListKa.begin(), tripletListKa.end());
	Kh.setFromTriplets(tripletListKh.begin(), tripletListKh.end());

	// 计算锚点贡献 Th
	//Th = Eigen::VectorXd::Zero(3);
	//Th(0) = mesh_d.get_vertex(MovedPoint).x();
	//Th(1) = mesh_d.get_vertex(MovedPoint).y();
	//Th(2) = mesh_d.get_vertex(MovedPoint).z();
	// 构建 Th 和 Ta
	Ta.resize(fixed.size() * 3);
	for (size_t i = 0; i < fixed.size(); ++i) {
		igame::Vertex p = mesh.get_vertex(igame::vHandle(fixed[i]));
		Ta.segment<3>(3 * i) = Eigen::Vector3d(p.x(), p.y(), p.z());
	}
	Th.resize(moved.size() * 3);
	for (size_t i = 0; i < moved.size(); ++i) {
		igame::Vertex p = mesh.get_vertex(igame::vHandle(moved[i]));
		Th.segment<3>(3 * i) = Eigen::Vector3d(p.x(), p.y(), p.z());
	}

	//标记固定点——固定点标记是为了做完矩阵乘法位置不变 
	for (auto& vh : fixed) {
		vis[fixed[vh]] = 1;
	}
	for (auto& vh : moved) {
		vis[moved[vh]] = 1;
	}
}

```
