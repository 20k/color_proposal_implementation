#include <iostream>

#include "color.hpp"
#include <assert.h>

#if 0
struct custom_color_space : color::basic_color_space<custom_color_space>
{
    float q=0, g=0, b=0, f=0, w=0;
};

struct custom_color : color::basic_color<custom_color_space>
{

};

///this function needs to optionally take a value, aka some sort of user specifiable black box
///eg in the case of SFML, we could pass in a renderwindow and make decisions based on gamma handling
///and have a custom colour type which is always correct regardless of the environment
void direct_convert(const custom_color& in, color::sRGB_float& test)
{

}
#endif // 0

struct P3_parameters
{
    static constexpr color::chromaticity R{0.68, 0.32};
    static constexpr color::chromaticity G{0.265, 0.69};
    static constexpr color::chromaticity B{0.15, 0.06};
    static constexpr color::chromaticity W = color::illuminant::CIE1931::D65;

    static constexpr float transfer_alpha = 1.055;
    static constexpr float transfer_beta = 0.0031308;
    static constexpr float transfer_gamma = 12/5.f;
    static constexpr float transfer_delta = 12.92;
    static constexpr float transfer_bdelta = 0.04045;

    static constexpr temporary::matrix_3x3 linear_to_XYZ = color::get_linear_RGB_to_XYZ(R, G, B, W);
    static constexpr temporary::matrix_3x3 XYZ_to_linear = linear_to_XYZ.invert();

    using gamma = color::transfer_function::default_parameterisation;
};

using P3_space = color::generic_RGB_space<P3_parameters>;
using P3_float = color::basic_color<P3_space, color::RGB_float_model>;

struct adobe_RGB_98_parameters
{
    static constexpr color::chromaticity R{0.64, 0.33};
    static constexpr color::chromaticity G{0.21, 0.71};
    static constexpr color::chromaticity B{0.15, 0.06};
    static constexpr color::chromaticity W = color::illuminant::CIE1931::D65;

    static constexpr float transfer_alpha = 1;
    static constexpr float transfer_beta = 0;
    static constexpr float transfer_gamma = 563/256.f;
    static constexpr float transfer_delta = 1;
    static constexpr float transfer_bdelta = 0;

    static constexpr temporary::matrix_3x3 linear_to_XYZ = color::get_linear_RGB_to_XYZ(R, G, B, W);
    static constexpr temporary::matrix_3x3 XYZ_to_linear = linear_to_XYZ.invert();

    using gamma = color::transfer_function::default_parameterisation;
};

using adobe_space = color::generic_RGB_space<adobe_RGB_98_parameters>;
using adobe_float = color::basic_color<adobe_space, color::RGB_float_model>;

int main()
{
    #if 0
    color::sRGB_uint8 val(255, 127, 80);
    color::sRGB_float fval;
    color::XYZ_float xyz_f(0, 0, 0);
    custom_color custom;

    color::convert(val, fval);

    assert(color::has_optimised_conversion(fval, xyz_f));
    assert(color::has_optimised_conversion(custom, fval));

    std::cout << "fval " << fval.r << " " << fval.g << " " << fval.b << std::endl;
    #endif // 0

    color::sRGB_uint8 val;
    val.r = 255;
    val.g = 127;
    val.b = 80;

    color::sRGB_float val2;

    ///performs a model conversion
    color::convert(val, val2);

    {
        color::sRGB_uint8 val3;
        val3.r = 255;
        val3.g = 127;
        val3.b = 80;

        color::XYZ test_xyz;

        static_assert(color::has_optimised_conversion(val3, test_xyz));

        color::convert(val3, test_xyz);

        color::sRGB_float val4;

        static_assert(color::has_optimised_conversion(test_xyz, val4));

        color::convert(test_xyz, val4);

        std::cout << "Roundtrip " << val4.r << " " << val4.g << " " << val4.b << std::endl;
    }

    {
        color::sRGBA_uint8 t1;
        t1.r = 255;
        t1.g = 127;
        t1.b = 80;
        t1.a = 230;

        color::sRGBA_float t2;

        color::convert(t1, t2);

        std::cout << "t2 " << t2.r << " " << t2.g << " " << t2.b << " " << t2.a << std::endl;
    }

    {
        color::sRGB_float t1;
        t1.r = 0;
        t1.g = 1;
        t1.b = 0;

        P3_float t2;

        static_assert(color::has_optimised_conversion(t1, t2));

        color::convert(t1, t2);

        std::cout << "P3 " << t2.r << " " << t2.g << " " << t2.b << std::endl;

        color::convert(t2, t1);

        std::cout << "sRGB " << t1.r << " " << t1.g << " " << t1.b << std::endl;
    }

    {
        color::sRGB_float t1;
        t1.r = 0;
        t1.g = 1;
        t1.b = 0;

        adobe_float t2;

        static_assert(color::has_optimised_conversion(t1, t2));

        color::convert(t1, t2);

        std::cout << "adobe " << t2.r << " " << t2.g << " " << t2.b << std::endl;
    }

    {
        adobe_float t1;
        t1.r = 1;
        t1.g = 0;
        t1.b = 1;

        color::XYZ t2;

        static_assert(color::has_optimised_conversion(t1, t2));

        color::convert(t1, t2);
    }

    {
        color::linear_RGB_float lin;
        lin.r = 1;
        lin.g = 0;
        lin.b = 1;

        color::sRGB_float srgb;

        static_assert(color::has_optimised_conversion(lin, srgb));

        color::convert(lin, srgb);

        std::cout << "CONVERTEDLIN " << srgb.r << " " << srgb.g << " " << srgb.b << std::endl;
    }

    {
        color::linear_RGB_float lin;
        lin.r = 1;
        lin.g = 0;
        lin.b = 1;

        color::sRGB_uint8 srgb;

        static_assert(color::has_optimised_conversion(lin, srgb));

        color::convert(lin, srgb);

        std::cout << "CONVERTEDLIN " << srgb.r << " " << srgb.g << " " << srgb.b << std::endl;
    }

    //color::basic_color<dummy> hello;

    return 0;
}
