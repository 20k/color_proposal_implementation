#pragma once

#include <color.hpp>

struct P3_parameters
{
    static constexpr color::chromaticity R{0.68, 0.32};
    static constexpr color::chromaticity G{0.265, 0.69};
    static constexpr color::chromaticity B{0.15, 0.06};
    static constexpr color::chromaticity W = color::illuminant::CIE1931::D65;

    static constexpr temporary::matrix_3x3 linear_to_XYZ = color::get_linear_RGB_to_XYZ(R, G, B, W);
    static constexpr temporary::matrix_3x3 XYZ_to_linear = linear_to_XYZ.invert();
};

struct P3_transfer_parameters
{
    static constexpr float transfer_alpha = 1.055;
    static constexpr float transfer_beta = 0.0031308;
    static constexpr float transfer_gamma = 12/5.f;
    static constexpr float transfer_delta = 12.92;
    static constexpr float transfer_bdelta = 0.04045;

    using gamma = color::transfer_function::default_parameterisation;
};

using P3_space = color::generic_RGB_space<P3_parameters, P3_transfer_parameters>;

struct P3_float : color::basic_color<P3_space, color::RGB_float_model, color::no_alpha>
{
    constexpr P3_float(float _r, float _g, float _b) {r = _r; g = _g; b = _b;}
    constexpr P3_float(){}
};

struct P3A_float : color::basic_color<P3_space, color::RGB_float_model, color::float_alpha>
{
    constexpr P3A_float(float _r, float _g, float _b, float _a) {r = _r; g = _g; b = _b; a = _a;}
    constexpr P3A_float(){}
};