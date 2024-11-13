
```
class APAP {
public:
	void first_solve();
	void preparation();
	//ARAP_Deformation(igame::VolumeMesh& _mesh, std::vector<igame::vHandle>& _fixed, Eigen::Vector3d& _pos) : mesh(_mesh), fixed(_fixed), pos(_pos) {};
	bool Exec();
	APAP(igame::VolumeMesh& _mesh, igame::VolumeMesh& _mesh_d, std::vector<igame::vHandle>& _fixed, std::vector<igame::vHandle>& _moved) : mesh(_mesh), resmesh(_mesh_d), fixed(_fixed), moved(_moved) {

	};

	void testDebug();
	void testDebug2(int i);
	void testPrint();


	void printSparseMatrix(const SpMat& mat, const std::string& name) {
		std::cout << "Matrix: " << name << " (Sparse)" << std::endl;
		for (int k = 0; k < mat.outerSize(); ++k) {
			for (SpMat::InnerIterator it(mat, k); it; ++it) {
				std::cout << "[" << it.row() << ", " << it.col() << "] = " << it.value() << std::endl;
			}
		}
		std::cout << "----------------------------------------" << std::endl;
	}

	void printDenseMatrix(const Eigen::MatrixXd& mat, const std::string& name) {
		std::cout << "Matrix: " << name << " (Dense)" << std::endl;
		for (int i = 0; i < mat.rows(); ++i) {
			for (int j = 0; j < mat.cols(); ++j) {
				std::cout << mat(i, j) << "\t";
			}
			std::cout << std::endl;
		}
		std::cout << "----------------------------------------" << std::endl;
	}

	void printVector(const Eigen::VectorXd& vec, const std::string& name) {
		std::cout << "Vector: " << name << std::endl;
		for (int i = 0; i < vec.size(); ++i) {
			std::cout << vec[i] << std::endl;
		}
		std::cout << "----------------------------------------" << std::endl;
	}

private:
	igame::VolumeMesh mesh;
	std::vector<Eigen::Vector3d> originVs;
	igame::VolumeMesh& resmesh;
	igame::VolumeMesh mesh_d;//变形后网格

	//算法里涉及到单个移动点的要改成vector形式的赋值，改完后再注释掉第27行
	igame::vHandle MovedPoint;

	std::vector<igame::vHandle>& fixed;
	std::vector<igame::vHandle>& moved;

	SpMat L, A, Kh, Ka, grad;
	Eigen::VectorXd V_star;
	//Eigen::MatrixXd V_star;
	Eigen::MatrixXd J;
	Eigen::VectorXd Th, Ta;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
	//Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	double lambda = 0.5;

	std::map<std::pair<igame::vHandle, igame::vHandle>, double> cots;
	std::map<std::pair<igame::vHandle, igame::vHandle>, Eigen::Vector3d> eij;
	std::vector<Eigen::Vector3d> pos_mesh_ref;
	std::map<igame::vHandle, int> vis;
};
```
