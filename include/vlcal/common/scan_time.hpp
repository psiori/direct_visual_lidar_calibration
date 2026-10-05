#pragma once

#include <optional>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace vlcal {

enum class TimeUnit { Seconds, Nanoseconds };

enum class TimeOrigin { FirstPoint, ScanStart };

enum class TimeConvention { Explicit, LegacyAuto };

struct ScanTimeParams {
  TimeUnit time_unit = TimeUnit::Seconds;
  TimeOrigin time_origin = TimeOrigin::FirstPoint;
  TimeConvention convention = TimeConvention::Explicit;
  std::optional<double> scan_duration;  ///< Full spin duration in seconds
};

struct ScanTimeError {
  bool ok = true;
  std::string message;
};

struct NormalizedScanTimes {
  std::vector<double> times;  ///< Seconds from chosen origin
  double scan_duration = 0.0;
};

/// Convert raw per-point times to seconds and apply origin / duration rules.
ScanTimeError normalize_scan_times(
  const std::vector<double>& raw_times,
  const ScanTimeParams& params,
  NormalizedScanTimes& out);

/// Alpha for pose interpolation: t / scan_duration.
inline double scan_time_alpha(double t, double scan_duration) {
  return t / std::max(1e-9, scan_duration);
}

}  // namespace vlcal
