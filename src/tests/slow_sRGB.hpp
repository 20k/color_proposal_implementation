#pragma once
#include <color.hpp>

struct slow_sRGB_space
{
    static inline constexpr temporary::matrix_3x3 impl_linear_to_XYZ = color::sRGB_parameters::linear_to_XYZ;
    static inline constexpr temporary::matrix_3x3 impl_XYZ_to_linear = color::sRGB_parameters::XYZ_to_linear;
};

struct slow_sRGB_model
{
    float r = 0;
    float g = 0;
    float b = 0;

    constexpr slow_sRGB_model(){}
    constexpr slow_sRGB_model(float _r, float _g, float _b)
    {
        r = _r;
        g = _g;
        b = _b;
    }
};

struct slow_linear_sRGBA_color : color::basic_color<slow_sRGB_space, slow_sRGB_model, color::float_alpha>
{
    constexpr slow_linear_sRGBA_color(float _r, float _g, float _b, float _a)
    {
        r = _r;
        g = _g;
        b = _b;
        a = _a;
    }

    constexpr slow_linear_sRGBA_color(){}
};

template<typename A1, typename A2>
constexpr inline
void color_convert(const color::basic_color<color::XYZ_space, color::XYZ_model, A1>& in, color::basic_color<slow_sRGB_space, slow_sRGB_model, A2>& out)
{
    auto transformed = temporary::multiply(slow_sRGB_space::impl_XYZ_to_linear, (temporary::vector_1x3){in.X, in.Y, in.Z});

    out.r = transformed.a[0];
    out.g = transformed.a[1];
    out.b = transformed.a[2];

    color::alpha_convert(in, out);
}

template<typename A1, typename A2>
constexpr inline
void color_convert(const color::basic_color<slow_sRGB_space, slow_sRGB_model, A1>& in, color::basic_color<color::XYZ_space, color::XYZ_model, A2>& out)
{
    auto transformed = temporary::multiply(slow_sRGB_space::impl_linear_to_XYZ, (temporary::vector_1x3){in.r, in.g, in.b});

    out.X = transformed.a[0];
    out.Y = transformed.a[1];
    out.Z = transformed.a[2];

    color::alpha_convert(in, out);
}
