#pragma once

#include <cstring>
#include <cstdlib>
#include <type_traits>
#include "rgb.h"
#include "vec.h"

/**
 * Convolve an RGB frame with a floating point matrix.
 *
 * @param frame the RGB frame
 * @param frame_dim Dimensions of the frame {X, Y}
 * @param mat the Matrix
 * @param mat_dim Dimensions of the matrix {X, Y}
 * @param mat_off Offset of the Matrix relative to the pixel to calculate.
 * For normal 3x3 Matrices this should be {-1, -1}
 * @return void. This function writes the result back into frame.
 */

template<typename ResType, typename MatType>
void convolve(const RGB* frame, Vec3D<ResType>* result, Vec2D<size_t> frame_dim, MatType* mat, Vec2D<size_t> mat_dim, Vec2D<int> mat_off) {
    const size_t buf_width = frame_dim.x + mat_dim.x - 1;
    const size_t buf_height = frame_dim.y + mat_dim.y - 1;
    // Move data to new buffer with 0s at edges.
    RGB* buf = (RGB*)calloc(buf_width * buf_height, sizeof(RGB));
    for (size_t i = 0; i < frame_dim.y; i++)
    {
        size_t dst_idx = (i - mat_off.y) * buf_width - mat_off.x;
        size_t src_idx = i * frame_dim.x;
        std::memcpy(&buf[dst_idx], &frame[src_idx], frame_dim.x * sizeof(RGB));
    }
    // Convolve
    for (size_t y = 0; y < frame_dim.y; y++)
    {
        for (size_t x = 0; x < frame_dim.x; x++)
        {
            typename std::common_type<ResType, MatType>::type r = 0, g = 0, b = 0;
            for (size_t v = 0; v < mat_dim.y; v++)
            {
                for (size_t u = 0; u < mat_dim.x; u++)
                {
                    MatType mat_val = mat[mat_dim.x * v + u];
                    size_t buf_idx = (y + v) * buf_width + (x + u);
                    r += mat_val * buf[buf_idx].r;
                    g += mat_val * buf[buf_idx].g;
                    b += mat_val * buf[buf_idx].b;
                }
            }
            size_t frame_idx = y * frame_dim.x + x;
            result[frame_idx].r = (ResType)r;
            result[frame_idx].g = (ResType)g;
            result[frame_idx].b = (ResType)b;
        }
    }
    free(buf);
}
