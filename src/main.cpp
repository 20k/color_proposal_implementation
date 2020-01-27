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

struct P3_float : color::basic_color<P3_space, color::RGB_float_model>
{
    constexpr P3_float(float _r, float _g, float _b) {r = _r; g = _g; b = _b;}
    constexpr P3_float(){}
};

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
//using adobe_float = color::basic_color<adobe_space, color::RGB_float_model>;

struct adobe_float : color::basic_color<adobe_space, color::RGB_float_model>
{
    constexpr adobe_float(float _r, float _g, float _b) {r = _r; g = _g; b = _b;}
    constexpr adobe_float(){}
};

struct weirdo_float_value
{
    using type = float;

    static inline constexpr float min = 0;
    static inline constexpr float max = 255;
};

struct weirdo_float_model : color::RGB_model<weirdo_float_value, weirdo_float_value, weirdo_float_value>
{

};

struct weirdo_linear_space : color::basic_color<color::linear_RGB_space, weirdo_float_model>
{
    constexpr weirdo_linear_space(float _r, float _g, float _b){r = _r; g = _g; b = _b;}
    constexpr weirdo_linear_space(){}
};

constexpr bool approx_equal(float v1, float v2)
{
    if(v1 < v2)
        return (v2 - v1) < 0.000001f;
    else
        return (v1 - v2) < 0.000001f;
}

void tests()
{
    {
        constexpr color::sRGB_float t1(0, 1, 0);

        static_assert(color::has_optimised_conversion(t1, P3_float()));

        constexpr P3_float t2 = color::convert<P3_float>(t1);

        static_assert(approx_equal(t2.r, 0.458407));
        static_assert(approx_equal(t2.g, 0.985265));
        static_assert(approx_equal(t2.b, 0.29832));
    }

    {
        constexpr weirdo_linear_space i_dislike_type_safety(255, 255, 128);

        static_assert(color::has_optimised_conversion(color::sRGB_float(), i_dislike_type_safety));
        static_assert(color::has_optimised_conversion(P3_float(), i_dislike_type_safety));
        static_assert(color::has_optimised_conversion(color::linear_RGB_float(), i_dislike_type_safety));

        constexpr color::linear_RGB_float linear = color::convert<color::linear_RGB_float>(i_dislike_type_safety);

        static_assert(approx_equal(linear.r, 1.f));
        static_assert(approx_equal(linear.g, 1.f));
        static_assert(approx_equal(linear.b, (128/255.f)));

        constexpr color::sRGB_uint8 srgb = color::convert<color::sRGB_uint8>(i_dislike_type_safety);

        static_assert(approx_equal(srgb.r, 255));
        static_assert(approx_equal(srgb.g, 255));
        static_assert(approx_equal(srgb.b, 188));
    }

    {
        color::sRGB_uint8 val3(255, 127, 80);

        color::XYZ test_xyz;

        static_assert(color::has_optimised_conversion(val3, test_xyz));

        color::convert(val3, test_xyz);

        color::sRGB_float val4;

        static_assert(color::has_optimised_conversion(test_xyz, val4));

        color::convert(test_xyz, val4);

        assert(approx_equal(val4.r, 1));
        assert(approx_equal(val4.g, 0.498039));
        assert(approx_equal(val4.b, 0.313725));
    }

    {
        color::sRGBA_uint8 t1(255, 127, 80, 230);

        color::sRGBA_float t2;

        color::convert(t1, t2);

        assert(approx_equal(t2.r, 1));
        assert(approx_equal(t2.g, 0.498039));
        assert(approx_equal(t2.b, 0.313726)); ///slightly different results to above due to imprecision
        assert(approx_equal(t2.a, 0.901961));
    }

    {
        color::sRGB_float t1(0, 1, 0);

        P3_float t2;

        static_assert(color::has_optimised_conversion(t1, t2));

        color::convert(t1, t2);

        assert(approx_equal(t2.r, 0.458407));
        assert(approx_equal(t2.g, 0.985265));
        assert(approx_equal(t2.b, 0.29832));

        color::convert(t2, t1);

        assert(approx_equal(t1.r, 0));
        assert(approx_equal(t1.g, 1));
        assert(approx_equal(t1.b, 0));
    }

    {
        color::sRGB_float t1(0, 1, 0);

        adobe_float t2;

        static_assert(color::has_optimised_conversion(t1, t2));

        color::convert(t1, t2);

        assert(approx_equal(t2.r, 0.564978));
        assert(approx_equal(t2.g, 1));
        assert(approx_equal(t2.b, 0.234443));
    }

    {
        adobe_float t1(1, 0, 1);

        color::XYZ t2;

        static_assert(color::has_optimised_conversion(t1, t2));

        color::convert(t1, t2);
    }

    {
        color::linear_RGB_float lin(1, 0, 1);

        color::sRGB_float srgb;

        static_assert(color::has_optimised_conversion(lin, srgb));

        color::convert(lin, srgb);

        assert(approx_equal(srgb.r, 1));
        assert(approx_equal(srgb.g, 0));
        assert(approx_equal(srgb.b, 1));
    }

    {
        color::linear_RGB_float lin(0.5, 1, 0.5);

        color::sRGB_uint8 srgb;

        static_assert(color::has_optimised_conversion(lin, srgb));

        color::convert(lin, srgb);

        assert(approx_equal(srgb.r, 188));
        assert(approx_equal(srgb.g, 255));
        assert(approx_equal(srgb.b, 188));
    }
}

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

    tests();

    //color::basic_color<dummy> hello;

    return 0;
}
