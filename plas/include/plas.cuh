#include "permuters.cuh"

enum BorderType
{
  REFLECT = 0,
  CIRCULAR = 1
};

std::pair<float*, float*>
sort_with_plas(const float* grid_input,
               int D,
               int H,
               int W,
               int min_block_size = 16,
               int min_blur_radius = 1,
               float radius_factor = 0.95,
               float improvement_break = 1e-5,
               int border_type_x = BorderType::REFLECT,
               int border_type_y = BorderType::REFLECT,
               int seed = 0,
               bool verbose = false,
               RandomPermuter* permute_config = new LCGPermuter());