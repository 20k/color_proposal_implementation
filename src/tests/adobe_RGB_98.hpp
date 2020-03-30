#pragma once

#include <color.hpp>

struct adobe_RGB_98_parameters
{
    static constexpr color::chromaticity R{0.64, 0.33};
    static constexpr color::chromaticity G{0.21, 0.71};
    static constexpr color::chromaticity B{0.15, 0.06};
    static constexpr color::chromaticity W = color::illuminant::CIE1931::D65;

    static constexpr temporary::matrix_3x3 linear_to_XYZ = color::get_linear_RGB_to_XYZ(R, G, B, W);
    static constexpr temporary::matrix_3x3 XYZ_to_linear = linear_to_XYZ.invert();
};

struct adobe_RGB_98_transfer_parameters
{
    static constexpr float transfer_alpha = 1;
    static constexpr float transfer_beta = 0;
    static constexpr float transfer_gamma = 563/256.f;
    static constexpr float transfer_delta = 1;
    static constexpr float transfer_bdelta = 0;

    template<typename T, typename U>
    using transfer_function = typename color::transfer_function<T, U>::gamma;
};

using adobe_space = color::generic_RGB_space<adobe_RGB_98_parameters, adobe_RGB_98_transfer_parameters>;

struct adobe_float : color::basic_color<adobe_space, color::RGB_float_model, color::no_alpha>
{
    constexpr adobe_float(float _r, float _g, float _b) {r = _r; g = _g; b = _b;}
    constexpr adobe_float(){}
};
