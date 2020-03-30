#ifndef RUNTIME_RGB_HPP_INCLUDED
#define RUNTIME_RGB_HPP_INCLUDED

#include <color.hpp>

struct runtime_RGB_space_tag{};

struct runtime_RGB_data
{
    temporary::matrix_3x3 impl_linear_to_XYZ = color::sRGB_parameters::linear_to_XYZ;
    temporary::matrix_3x3 impl_XYZ_to_linear = color::sRGB_parameters::XYZ_to_linear;
};

struct runtime_RGB_color : color::basic_color<runtime_RGB_space_tag, color::RGB_float_model, color::no_alpha>
{
    constexpr runtime_RGB_color(float _r, float _g, float _b)
    {
        r = _r;
        g = _g;
        b = _b;
    }

    constexpr runtime_RGB_color(){}
};

template<typename A1, typename A2>
constexpr inline
void color_convert(const color::basic_color<color::XYZ_space, color::XYZ_model, A1>& in, color::basic_color<runtime_RGB_space_tag, color::RGB_float_model, A2>& out, const runtime_RGB_data& data)
{
    auto vec = temporary::multiply(data.impl_XYZ_to_linear, temporary::vector_1x3{in.X, in.Y, in.Z});

    out.r = color::RGB_float_model::R_transfer::gamma::from_linear(vec.a[0], color::sRGB_parameters());
    out.g = color::RGB_float_model::G_transfer::gamma::from_linear(vec.a[1], color::sRGB_parameters());
    out.b = color::RGB_float_model::B_transfer::gamma::from_linear(vec.a[2], color::sRGB_parameters());

    color::alpha_convert(in, out);
}

template<typename A1, typename A2>
constexpr inline
void color_convert(const color::basic_color<runtime_RGB_space_tag, color::RGB_float_model, A1>& in, color::basic_color<color::XYZ_space, color::XYZ_model, A2>& out, const runtime_RGB_data& data)
{
    auto lin_r = color::RGB_float_model::R_transfer::gamma::to_linear(in.r, color::sRGB_parameters());
    auto lin_g = color::RGB_float_model::G_transfer::gamma::to_linear(in.g, color::sRGB_parameters());
    auto lin_b = color::RGB_float_model::B_transfer::gamma::to_linear(in.b, color::sRGB_parameters());

    auto vec = temporary::multiply(data.impl_linear_to_XYZ, temporary::vector_1x3{lin_r.v, lin_g.v, lin_b.v});

    out.X = vec.a[0];
    out.Y = vec.a[1];
    out.Z = vec.a[2];

    color::alpha_convert(in, out);
}


#endif // RUNTIME_RGB_HPP_INCLUDED
