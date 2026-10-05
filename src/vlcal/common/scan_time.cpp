#include <vlcal/common/scan_time.hpp>

#include <algorithm>
#include <cmath>

#include <vlcal/common/time_keeper.hpp>

namespace vlcal {

namespace {

std::vector<double> to_seconds(const std::vector<double>& raw, TimeUnit unit) {
  std::vector<double> times(raw.size());
  const double scale = (unit == TimeUnit::Nanoseconds) ? 1e-9 : 1.0;
  for (size_t i = 0; i < raw.size(); ++i) {
    times[i] = raw[i] * scale;
  }
  return times;
}

}  // namespace

ScanTimeError normalize_scan_times(const std::vector<double>& raw_times, const ScanTimeParams& params, NormalizedScanTimes& out) {
  out = NormalizedScanTimes{};
  if (raw_times.empty()) {
    return {false, "empty per-point times"};
  }

  if (params.convention == TimeConvention::LegacyAuto) {
    RawPoints::Ptr raw = std::make_shared<RawPoints>();
    raw->stamp = 0.0;
    raw->times = to_seconds(raw_times, params.time_unit);
    TimeKeeper keeper(TimeKeeper::legacy_auto_params());
    keeper.replace_points_stamp(raw);
    out.times = raw->times;
    out.scan_duration = out.times.back();
    if (params.scan_duration.has_value()) {
      out.scan_duration = *params.scan_duration;
    }
    return {};
  }

  std::vector<double> times = to_seconds(raw_times, params.time_unit);

  if (params.time_origin == TimeOrigin::FirstPoint) {
    const double t0 = *std::min_element(times.begin(), times.end());
    for (auto& t : times) {
      t -= t0;
    }
    out.scan_duration = times.back();
    if (params.scan_duration.has_value()) {
      out.scan_duration = *params.scan_duration;
    }
  } else {
    if (!params.scan_duration.has_value() || *params.scan_duration <= 0.0) {
      return {false, "scan_duration required for TimeOrigin::ScanStart"};
    }
    out.scan_duration = *params.scan_duration;
  }

  if (out.scan_duration <= 0.0) {
    return {false, "invalid scan_duration"};
  }

  out.times = std::move(times);
  return {};
}

}  // namespace vlcal
