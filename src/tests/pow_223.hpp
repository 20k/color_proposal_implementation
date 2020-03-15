#pragma once

#include <color.hpp>

struct sRGB_approx_transfer_parameters
{
    static constexpr float transfer_alpha = 1;
    static constexpr float transfer_beta = 0;
    static constexpr float transfer_gamma = 563/256.f;
    static constexpr float transfer_delta = 1;
    static constexpr float transfer_bdelta = 0;

    using gamma = color::transfer_function::default_parameterisation;
};

using approx_sRGB_space = color::generic_RGB_space<color::sRGB_parameters, sRGB_approx_transfer_parameters>;
using approx_sRGB_uint8 = color::basic_color<approx_sRGB_space, color::RGB_uint8_model, color::no_alpha>;
