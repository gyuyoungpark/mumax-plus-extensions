target_sources(physics PRIVATE
    src/afmtype.cu
    src/afmtype.hpp
)

target_include_directories(physics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
