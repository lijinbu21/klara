```
void ModelDraw::MoveCtr3D()
{
	std::vector<Eigen::Vector3d> targetCtrPoints3D;
	igame::VolumeMesh* CtrMesh;
	igame::VolumeMesh* RealMesh, * DeformMesh;
	for (auto& dm : drawableVolumeMeshes) {
		if (dm->mesh->isCtrMesh)
			CtrMesh = dm->mesh;
		else
		{
			RealMesh = dm->mesh;
			DeformMesh = dm->mesh;
		}
	}
	for (int i = 0; i < CtrMesh->vsize(); ++i) {
		auto& p = CtrMesh->get_pos(igame::vHandle(i));
		//auto& v = (p - dm->render_center) * dm->render_scale;
		//targetCtrPoints3D.emplace_back(v.x(), v.y(), v.z());
		targetCtrPoints3D.emplace_back(p.x(), p.y(), p.z());
		std::cout << targetCtrPoints3D[i].transpose() << std::endl;
	}
	ModelDrawDeform3D.loadTargetMesh(targetCtrPoints3D);
	ModelDrawDeform3D.recompute();

	for (int i = 0; i < DeformMesh->vsize(); ++i) {
		auto& p = DeformMesh->get_vertex(igame::vHandle(i));
		p.pos() = igame::vec3{ ModelDrawDeform3D.newPoints[i].x(),ModelDrawDeform3D.newPoints[i].y(),ModelDrawDeform3D.newPoints[i].z() };
		//p.pos() = p.pos() / dm->render_scale + dm->render_center;
	}

	//ARAP_Deformation tmp(*RealMesh, ModelDrawDeform3D.fixed_vh, ModelDrawDeform3D.newPoints[ModelDrawDeform3D.moved_vh]);
	APAP tmp(*RealMesh, *DeformMesh, ModelDrawDeform3D.fixed_vh, ModelDrawDeform3D.moved_vh);
	printf("APAP_Deformation Executing..\n");
	tmp.Exec();
	printf("APAP_Deformation Executed!\n");

	for (int i = 0; i < RealMesh->vsize(); ++i) {
		auto& p = RealMesh->get_vertex(igame::vHandle(i));
		cout << i << " : (" << p.x() << "," << p.y() << "," << p.z() << ")" << endl;
		//p.pos() = p.pos() / dm->render_scale + dm->render_center;
	}

	//for (auto& dm : drawableVolumeMeshes) {
	//	dm->UpdateGL();
	//}

	//for (auto& dm : drawableVolumeMeshes) {
	//	if (dm->mesh->isCtrMesh) {
	//		for (int i = 0; i < dm->mesh->vsize(); ++i) {
	//			auto& p = dm->mesh->get_pos(igame::vHandle(i));
	//			//auto& v = (p - dm->render_center) * dm->render_scale;
	//			//targetCtrPoints3D.emplace_back(v.x(), v.y(), v.z());
	//			targetCtrPoints3D.emplace_back(p.x(), p.y(), p.z());
	//			std::cout << targetCtrPoints3D[i].transpose() << std::endl;
	//		}
	//		ModelDrawDeform3D.loadTargetMesh(targetCtrPoints3D);
	//		ModelDrawDeform3D.recompute();
	//		dm->UpdateGL();
	//		break;
	//	}
	//}
}
```