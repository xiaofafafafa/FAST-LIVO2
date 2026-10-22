# Shared by the catkin build and the dependency-light standalone test build.
add_library(adaptive_window_core
  src/adaptive_window/imu_preintegration.cpp
  src/adaptive_window/information_evaluator.cpp
  src/adaptive_window/adaptive_policy.cpp
  src/adaptive_window/temporal_associator.cpp
  src/adaptive_window/fixed_lag_optimizer.cpp
  src/adaptive_window/factor_builders.cpp
  src/adaptive_window/adaptive_window_backend.cpp
)
target_include_directories(adaptive_window_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(adaptive_window_core PUBLIC Eigen3::Eigen Ceres::ceres)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # The frontend's ARM flags must not invalidate finite/NaN safety checks.
  target_compile_options(adaptive_window_core PRIVATE -fno-fast-math)
endif()

function(add_adaptive_window_tests)
  add_executable(adaptive_window_smoke test/adaptive_window_smoke.cpp)
  target_link_libraries(adaptive_window_smoke PRIVATE adaptive_window_core)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(adaptive_window_smoke PRIVATE -fno-fast-math)
  endif()
  add_test(NAME adaptive_window_smoke COMMAND adaptive_window_smoke)
endfunction()
