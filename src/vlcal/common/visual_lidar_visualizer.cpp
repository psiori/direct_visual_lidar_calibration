#include <vlcal/common/visual_lidar_visualizer.hpp>

#include <guik/viewer/light_viewer.hpp>

namespace vlcal {

VisualLiDARVisualizer::VisualLiDARVisualizer(const camera::GenericCameraBase::ConstPtr& proj, const std::vector<VisualLiDARData::ConstPtr>& dataset, const bool draw_sphere)
: draw_sphere(draw_sphere),
  proj(proj),
  dataset(dataset),
  T_camera_lidar(Eigen::Isometry3d::Identity()) {
  //
  auto viewer = guik::LightViewer::instance();
  viewer->set_draw_xy_grid(false);
  viewer->use_arcball_camera_control();

  selected_bag_id = -1;
  blend_weight = 0.7f;
  viewer->register_ui_callback("ui", [this] { ui_callback(); });

  kill_switch = false;
  color_update_thread = std::thread([this] { color_update_task(); });
}

VisualLiDARVisualizer::~VisualLiDARVisualizer() {
  kill_switch = true;
  color_update_thread.join();
}

void VisualLiDARVisualizer::ui_callback() {
  auto viewer = guik::LightViewer::instance();

  ImGui::Begin("visualizer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  const int prev_selected_bag_id = selected_bag_id;
  if (ImGui::ArrowButton("##Left", ImGuiDir_Left)) {
    selected_bag_id--;
  }
  ImGui::SameLine();
  if (ImGui::ArrowButton("##Right", ImGuiDir_Right)) {
    selected_bag_id++;
  }
  ImGui::SameLine();
  ImGui::DragInt("bag_id", &selected_bag_id, 1, 0, dataset.size() - 1);
  selected_bag_id = std::max<int>(0, std::min<int>(dataset.size() - 1, selected_bag_id));

  if (prev_selected_bag_id != selected_bag_id) {
    std::lock_guard<std::mutex> lock(updater_mutex);
    color_updater.reset(new PointsColorUpdater(proj, dataset[selected_bag_id]->image, dataset[selected_bag_id]->points));
    viewer->update_drawable("points", color_updater->cloud_buffer, guik::VertexColor());

    if (draw_sphere) {
      sphere_updater.reset(new PointsColorUpdater(proj, dataset[selected_bag_id]->image));
      viewer->update_drawable("sphere", sphere_updater->cloud_buffer, guik::VertexColor());
    }
  }
  ImGui::DragFloat("blend_weight", &blend_weight, 0.01f, 0.0f, 1.0f);

  ImGui::End();
}

void VisualLiDARVisualizer::set_T_camera_lidar(const Eigen::Isometry3d& T_camera_lidar) {
  this->T_camera_lidar = T_camera_lidar;
}

bool VisualLiDARVisualizer::spin_once() {
  auto viewer = guik::LightViewer::instance();
  return viewer->spin_once();
}

void VisualLiDARVisualizer::color_update_task() {
  while (!kill_switch) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::lock_guard<std::mutex> lock(updater_mutex);
    if (color_updater) {
      color_updater->update(T_camera_lidar, blend_weight);
    }
    if (sphere_updater) {
      sphere_updater->update(T_camera_lidar, 1.0);
    }
  }
}

}  // namespace vlcal