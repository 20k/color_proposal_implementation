#pragma once

#include <color.hpp>

struct weirdo_float_value
{
    using type = float;

    static inline constexpr float base = 0;
    static inline constexpr float scale = 255;

    static inline constexpr float min = 0;
    static inline constexpr float max = 255;
};

struct weirdo_float_model : color::RGB_model<weirdo_float_value, weirdo_float_value, weirdo_float_value>
{

};

struct weirdo_linear_space : color::basic_color<color::linear_sRGB_space, weirdo_float_model, color::no_alpha>
{
    constexpr weirdo_linear_space(float _r, float _g, float _b){r = _r; g = _g; b = _b;}
    constexpr weirdo_linear_space(){}
};