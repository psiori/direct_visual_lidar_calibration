#include <vlcal/common/visual_lidar_data_loader.hpp>

#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <vlcal/common/console_colors.hpp>

#include <glk/io/ply_io.hpp>

namespace vlcal {

VisualLiDARData::Ptr load_visual_lidar_data(const std::string& data_path, const std::string& bag_name) {
  std::cout << "loading " << data_path + "/" + bag_name + ".(png|ply)" << std::endl;

  auto data = std::make_shared<VisualLiDARData>();
  data->image = cv::imread(data_path + "/" + bag_name + ".png", 0);
  if (!data->image.data) {
    std::cerr << vlcal::console::bold_red << "warning: failed to load " << data_path + "/" + bag_name + ".png" << vlcal::console::reset << std::endl;
    abort();
  }

  auto ply = glk::load_ply(data_path + "/" + bag_name + ".ply");
  if (!ply) {
    std::cerr << vlcal::console::bold_red << "warning: failed to load " << data_path + "/" + bag_name + ".ply" << vlcal::console::reset << std::endl;
    abort();
  }

  data->points = std::make_shared<FrameCPU>(ply->vertices);
  data->points->add_intensities(ply->intensities);
  return data;
}

}  // namespace vlcal
