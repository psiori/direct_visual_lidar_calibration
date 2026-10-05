from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy
import os


class DirectVisualLidarCalibrationConan(ConanFile):
    name = "vlcal_align"
    version = "0.1.0"
    license = "MIT"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "build_vlcal_preprocess": [True, False],
        "build_with_viewer": [True, False],
        "build_with_march_native": [True, False],
    }
    default_options = {
        "shared": True,
        "fPIC": True,
        "build_vlcal_preprocess": True,
        "build_with_viewer": False,
        "build_with_march_native": True,
    }
    exports_sources = "*"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        if self.options.shared:
            self.options.rm_safe("fPIC")

    def requirements(self):
        self.requires("eigen/3.4.0")
        self.requires("ceres-solver/2.2.0")
        self.requires("opencv/4.10.0")
        if self.options.build_vlcal_preprocess:
            self.requires("gtsam/4.3a1")
            self.requires("pcl/1.14.1")
            self.requires("fmt/10.2.1")
            self.requires("boost/1.83.0")

    def layout(self):
        cmake_layout(self)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["BUILD_VLCAL_ALIGN"] = True
        tc.variables["BUILD_VLCAL_PREPROCESS"] = self.options.build_vlcal_preprocess
        tc.variables["BUILD_WITH_VIEWER"] = self.options.build_with_viewer
        tc.variables["BUILD_WITH_MARCH_NATIVE"] = self.options.build_with_march_native
        tc.variables["BUILD_VLCAL_TESTS"] = False
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "direct_visual_lidar_calibration")
        self.cpp_info.set_property("cmake_target_name", "direct_visual_lidar_calibration::vlcal_align")
        if self.options.build_vlcal_preprocess:
            self.cpp_info.components["align"].set_property("cmake_target_name", "direct_visual_lidar_calibration::vlcal_align")
            self.cpp_info.components["preprocess"].set_property("cmake_target_name", "direct_visual_lidar_calibration::vlcal_preprocess")
            self.cpp_info.components["preprocess"].requires = ["align"]
