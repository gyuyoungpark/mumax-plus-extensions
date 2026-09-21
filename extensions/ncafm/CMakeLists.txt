target_sources(physics PRIVATE
    src/oct_k6.cu
    src/oct_k6.hpp
    src/octupole.cu
    src/octupole.hpp
)

target_include_directories(physics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
