# CMakeLists.txt for heat diffusion example
# Location: ~/gyselalibxx/simulations/example/CMakeLists.txt

add_executable(heat_diffusion_1d
    main.cpp
)

target_link_libraries(heat_diffusion_1d
    PUBLIC
        DDC::core
        gslx::utils
)

# Include Gyselalibxx's utilities and aliases
target_include_directories(heat_diffusion_1d
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/utils
)

# Set C++ standard
target_compile_features(heat_diffusion_1d PUBLIC cxx_std_17)
