#include <vlcal/common/estimate_fov.hpp>

#include <algorithm>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/convex_hull.h>

namespace vlcal {

double estimate_lidar_fov(const Frame::ConstPtr& points) {
  auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  cloud->resize(points->size());
  std::transform(points->points, points->points + points->size(), cloud->begin(), [](const auto& p) { return pcl::PointXYZ(p.x(), p.y(), p.z()); });

  auto filtered = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.2f, 0.2f, 0.2f);
  voxelgrid.setInputCloud(cloud);
  voxelgrid.filter(*filtered);

  const auto remove_loc = std::remove_if(filtered->begin(), filtered->end(), [](const pcl::PointXYZ& pt) { return pt.getVector3fMap().norm() < 1.0; });
  filtered->erase(remove_loc, filtered->end());
  cloud = filtered;

  pcl::ConvexHull<pcl::PointXYZ> convexhull;
  convexhull.setInputCloud(cloud);

  auto hull = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  convexhull.reconstruct(*hull);

  std::vector<Eigen::Vector3d> dirs(hull->size());
  for (int i = 0; i < hull->size(); i++) {
    dirs[i] = hull->at(i).getVector3fMap().cast<double>().normalized();
  }

  double min_cosine = M_PI;
  for (int i = 0; i < hull->size(); i++) {
    for (int j = i + 1; j < hull->size(); j++) {
      const double cosine = dirs[i].dot(dirs[j]);
      min_cosine = std::min(cosine, min_cosine);
    }
  }

  return std::acos(min_cosine);
}

}  // namespace vlcal
