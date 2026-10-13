#include <vlcal/preprocess/dynamic_point_cloud_integrator.hpp>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <vlcal/common/ivox.hpp>
#include <vlcal/common/kdtree2.hpp>
#include <vlcal/common/frame_cpu.hpp>
#include <vlcal/common/integrated_ct_gicp_factor.hpp>
#include <vlcal/common/vector3i_hash.hpp>

#include <vlcal/common/cloud_covariance_estimation.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>

#ifdef VLCAL_WITH_IRIDESCENCE
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>
#endif

namespace vlcal {

namespace {

void log_integrator_diagnostics(
  int scan_index,
  bool is_seed,
  const Frame& frame,
  double scan_duration,
  int subsample_points,
  std::size_t target_map_points,
  std::size_t output_voxels,
  const gtsam::Pose3& begin,
  const gtsam::Pose3& end,
  double lm_error_initial,
  double lm_error_final) {
  double time_min = 0.0;
  double time_max = 0.0;
  if (frame.has_times() && frame.size() > 0) {
    const auto [tmin, tmax] =
      std::minmax_element(frame.times, frame.times + frame.size());
    time_min = *tmin;
    time_max = *tmax;
  }
  const double inv_dur = 1.0 / std::max(1e-9, scan_duration);
  const gtsam::Pose3 delta = begin.between(end);
  const double scan_trans = delta.translation().norm();
  const gtsam::Vector3 scan_omega = gtsam::Rot3::Logmap(delta.rotation());
  const double scan_rot_deg = scan_omega.norm() * (180.0 / M_PI);
  const double odom_trans = end.translation().norm();
  const gtsam::Vector3 odom_omega = gtsam::Rot3::Logmap(end.rotation());
  const double odom_rot_deg = odom_omega.norm() * (180.0 / M_PI);

  std::cerr << "[integrator-diag] scan=" << scan_index << (is_seed ? " (seed)" : "") << "\n"
            << "  raw_points=" << frame.size() << " subsample=" << subsample_points
            << " target_map_pts=" << target_map_points << " output_voxels=" << output_voxels << "\n"
            << std::fixed << std::setprecision(6) << "  time_s=[" << time_min << ", " << time_max << "] alpha=["
            << time_min * inv_dur << ", " << time_max * inv_dur << "]\n"
            << "  lm_error initial=" << lm_error_initial << " final=" << lm_error_final << "\n"
            << std::setprecision(4) << "  scan_motion trans_m=" << scan_trans << " rot_deg=" << scan_rot_deg << "\n"
            << "  odom_end trans_m=" << odom_trans << " rot_deg=" << odom_rot_deg << "\n";
}

}  // namespace

DynamicPointCloudIntegratorParams::DynamicPointCloudIntegratorParams() {
  visualize = false;
  verbose = false;
  integrator_debug = false;
  num_threads = 16;
  k_neighbors = 20;
  target_num_points = 10000;
  voxel_resolution = 0.05;
  min_distance = 1.0;
}

DynamicPointCloudIntegratorParams::~DynamicPointCloudIntegratorParams() {
}

DynamicPointCloudIntegrator::DynamicPointCloudIntegrator(const DynamicPointCloudIntegratorParams& params) : params(params) {
  last_T_odom_lidar_begin = gtsam::Pose3();
  last_T_odom_lidar_end = gtsam::Pose3();

  // Note: voxel resolution is the container leaf resolution for the iVox search
  // structure and not related to DynamicPointCloudIntegratorParams::voxel_resolution.
  // Likewise, the insertion distance threshold is not
  // DynamicPointCloudIntegratorParams::min_distance
  target_ivox.reset(new iVox(1, 0.05, 100));

  voxelgrid_thread = std::thread([this] { voxelgrid_task(); });
}

DynamicPointCloudIntegrator::~DynamicPointCloudIntegrator() {
  alignment_results.submit_end_of_data();
  if (voxelgrid_thread.joinable()) {
    voxelgrid_thread.join();
  }
}

void DynamicPointCloudIntegrator::insert_points(const Frame::ConstPtr& frame) {
  ++scan_count_;
  const int scan_index = scan_count_;
  auto frame_sorted = sort_by_time(frame);
  auto points = randomgrid_sampling(frame_sorted, 0.5, static_cast<double>(params.target_num_points) / frame_sorted->size(), mt);

  std::vector<int> neighbors(points->size() * params.k_neighbors);

  // Find kNN
  KdTree2<Frame> tree(points);
#pragma omp parallel for num_threads(params.num_threads)
  for (int i = 0; i < points->size(); i++) {
    std::vector<size_t> k_indices(params.k_neighbors);
    std::vector<double> k_sq_dists(params.k_neighbors);
    tree.knn_search(points->points[i].data(), params.k_neighbors, k_indices.data(), k_sq_dists.data());
    std::copy(k_indices.begin(), k_indices.end(), neighbors.begin() + params.k_neighbors * i);
  }

  // Estimate covariances
  CloudCovarianceEstimation covariance_estimation(params.num_threads);
  points->add_covs(covariance_estimation.estimate(points->points_storage, neighbors));

  if (!target_ivox->has_points()) {
    // Handling the first frame
    target_ivox->insert(*points);
    alignment_results.push(std::make_tuple(frame_sorted, gtsam::Pose3(), gtsam::Pose3()));
    if (params.integrator_debug) {
      const double scan_duration =
        frame_sorted->scan_duration > 0 ? frame_sorted->scan_duration : frame_sorted->times[frame_sorted->size() - 1];
      log_integrator_diagnostics(
        scan_index,
        true,
        *frame_sorted,
        scan_duration,
        static_cast<int>(points->size()),
        target_ivox->voxel_points().size(),
        voxelgrid.size(),
        gtsam::Pose3(),
        gtsam::Pose3(),
        0.0,
        0.0);
    }
    return;
  }

  const double scan_duration = frame_sorted->scan_duration > 0 ? frame_sorted->scan_duration : frame_sorted->times[frame_sorted->size() - 1];
  const gtsam::Vector6 pred_v_odom_lidar = gtsam::Pose3::Logmap(last_T_odom_lidar_begin.between(last_T_odom_lidar_end)) / scan_duration;
  const gtsam::Pose3 pred_T_odom_lidar_begin = last_T_odom_lidar_end;
  const gtsam::Pose3 pred_T_odom_lidar_end = last_T_odom_lidar_end * gtsam::Pose3::Expmap(0.75 * pred_v_odom_lidar * scan_duration);

  gtsam::Values values;
  values.insert(0, pred_T_odom_lidar_begin);
  values.insert(1, pred_T_odom_lidar_end);

  gtsam::NonlinearFactorGraph graph;
  auto factor = std::make_shared<IntegratedCT_GICPFactor_<iVox, Frame>>(0, 1, target_ivox, points, target_ivox);
  factor->set_num_threads(params.num_threads);
  if (frame_sorted->scan_duration > 0) {
    factor->set_time_span(0.0, frame_sorted->scan_duration);
  }
  graph.add(factor);

  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(0, gtsam::Pose3(pred_T_odom_lidar_begin), gtsam::noiseModel::Isotropic::Precision(6, 1e3));
  graph.emplace_shared<gtsam::BetweenFactor<gtsam::Pose3>>(0, 1, gtsam::Pose3(), gtsam::noiseModel::Isotropic::Precision(6, 1e5));

  gtsam::LevenbergMarquardtParams lm_params;
  lm_params.setMaxIterations(10);
  if (params.verbose) {
    lm_params.setVerbosityLM("SUMMARY");
  }

  const double lm_error_initial = graph.error(values);
  for (int i = 0; i < 3; i++) {
    values = gtsam::LevenbergMarquardtOptimizer(graph, values, lm_params).optimize();
  }
  const double lm_error_final = graph.error(values);

  last_T_odom_lidar_begin = values.at<gtsam::Pose3>(0);
  last_T_odom_lidar_end = values.at<gtsam::Pose3>(1);

  auto deskewed = std::make_shared<FrameCPU>(factor->deskewed_source_points(values));
  deskewed->add_covs(covariance_estimation.estimate(deskewed->points_storage, neighbors));
  deskewed->add_intensities(points->intensities, points->size());
  target_ivox->insert(*deskewed);

  alignment_results.push(std::make_tuple(frame_sorted, values.at<gtsam::Pose3>(0), values.at<gtsam::Pose3>(1)));

  if (params.integrator_debug) {
    log_integrator_diagnostics(
      scan_index,
      false,
      *frame_sorted,
      scan_duration,
      static_cast<int>(points->size()),
      target_ivox->voxel_points().size(),
      voxelgrid.size(),
      last_T_odom_lidar_begin,
      last_T_odom_lidar_end,
      lm_error_initial,
      lm_error_final);
  }

#ifdef VLCAL_WITH_IRIDESCENCE
  if (params.visualize) {
    auto viewer = guik::LightViewer::instance();
    viewer->update_drawable("coord", glk::Primitives::coordinate_system(), guik::VertexColor(last_T_odom_lidar_end.matrix()));
    viewer->update_drawable("points", std::make_shared<glk::PointCloudBuffer>(deskewed->points, deskewed->size()), guik::FlatOrange());
    viewer->update_drawable("target", std::make_shared<glk::PointCloudBuffer>(target_ivox->voxel_points()), guik::Rainbow());
    viewer->spin_once();
  }
#endif
}

void DynamicPointCloudIntegrator::voxelgrid_task() {
  while (true) {
    auto data = alignment_results.pop_lock();
    if (!data) {
      break;
    }

    const auto& raw_points = std::get<0>(*data);
    const auto& T_odom_lidar_begin = std::get<1>(*data);
    const auto& T_odom_lidar_end = std::get<2>(*data);
    const double scan_duration = raw_points->scan_duration > 0 ? raw_points->scan_duration : raw_points->times[raw_points->size() - 1];

    double last_t = -1.0;
    gtsam::Pose3 T_odom_lidar = T_odom_lidar_begin;

    const double time_eps = 1e-4;
    for (int i = 0; i < raw_points->size(); i++) {
      const double t = raw_points->times[i] / std::max(1e-9, scan_duration);

      if (t - last_t > time_eps) {
        last_t = t;
        T_odom_lidar = T_odom_lidar_begin.interpolateRt(T_odom_lidar_end, t);
      }

      if (raw_points->points[i].head<3>().norm() < params.min_distance) {
        continue;
      }

      const Eigen::Vector4d pt = T_odom_lidar.matrix() * raw_points->points[i];

      const Eigen::Vector3i coord = (pt / params.voxel_resolution).array().floor().cast<int>().head<3>();
      mtx_.lock();
      voxelgrid[coord] = Eigen::Vector4d(pt[0], pt[1], pt[2], raw_points->intensities[i]);
      mtx_.unlock();
    }

    /*
    std::vector<Eigen::Vector3f> points;
    std::vector<float> intensities;

    voxelgrid->forEachCell([&](const Eigen::Vector4d& value, const auto& coord) {
      points.emplace_back(value.head<3>().cast<float>());
      intensities.emplace_back(value.w());
    });

    auto viewer = guik::LightViewer::instance();
    auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(points);
    cloud_buffer->add_intensity(glk::COLORMAP::TURBO, intensities, 1.0f / (*std::max_element(intensities.begin(), intensities.end())));
    viewer->update_drawable("points", cloud_buffer, guik::VertexColor());
    viewer->spin_once();
    */
  }
}

Frame::ConstPtr DynamicPointCloudIntegrator::get_points() {
  // alignment_results.submit_end_of_data();
  // voxelgrid_thread.join();

  std::vector<Eigen::Vector3f> points;
  std::vector<float> intensities;

  mtx_.lock();
  points.reserve(voxelgrid.size());
  intensities.reserve(voxelgrid.size());

  for (const auto& voxel : voxelgrid) {
    points.emplace_back(voxel.second.cast<float>().head<3>());
    intensities.emplace_back(voxel.second.w());
  }
  mtx_.unlock();

  auto frame = std::make_shared<FrameCPU>(points);
  frame->add_intensities(intensities);
  return frame;
}

}  // namespace vlcal
