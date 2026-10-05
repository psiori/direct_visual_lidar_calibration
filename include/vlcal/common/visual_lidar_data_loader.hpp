#pragma once

#include <string>

#include <vlcal/common/visual_lidar_data.hpp>

namespace vlcal {

/// Load image + PLY pair from a dataset directory (viewer / CLI tools).
VisualLiDARData::Ptr load_visual_lidar_data(const std::string& data_path, const std::string& bag_name);

}  // namespace vlcal
