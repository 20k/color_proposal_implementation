#include <iostream>

#include "color.hpp"
#include <assert.h>
#include <algorithm>
#include <tuple>

#ifndef NO_TESTS
#include "tests/adobe_RGB_98.hpp"
#include "tests/common.hpp"
#include "tests/P3.hpp"
#include "tests/pow_223.hpp"
#include "tests/HSL.hpp"
#include "tests/0_255_linear.hpp"

struct retained_type : color::basic_color<color::linear_sRGB_space, color::RGB_float_model, color::no_alpha>
{
    constexpr retained_type(float _r, float _g, float _b)
    {
        r = _r;
        g = _g;
        b = _b;
    }

    constexpr retained_type()
    {

    }
};

template<typename space_1, typename model_1, typename gamma_1, typename alpha_1>
void color_convert(const retained_type& type, color::basic_color<color::generic_RGB_space<space_1, gamma_1>, model_1, alpha_1>& out)
{
    auto converted = temporary::multiply(retained_type::space_type::RGB_parameters::linear_to_XYZ, temporary::vector_1x3{type.r, type.g, type.b});

    out.X = converted.a[0];
    out.Y = converted.a[1];
    out.Z = converted.a[2];
}

struct fully_custom_colorspace
{
    float value = 0;
    float value2 = 0;

    ///Not necessary, purely for testing purposes. Not an RGB colorspace!
    static inline constexpr temporary::matrix_3x3 impl_linear_to_XYZ = color::sRGB_parameters::linear_to_XYZ;
    static inline constexpr temporary::matrix_3x3 impl_XYZ_to_linear = color::sRGB_parameters::XYZ_to_linear;
};

struct fully_custom_color : color::basic_color<fully_custom_colorspace, color::RGB_float_model, color::no_alpha>
{
    constexpr fully_custom_color(float _r, float _g, float _b){r = _r; g = _g; b = _b;}
    constexpr fully_custom_color(){}
};

void color_convert(const color::basic_color<color::XYZ_space, color::XYZ_model, color::no_alpha>& in, color::basic_color<fully_custom_colorspace, color::RGB_float_model, color::no_alpha>& out, fully_custom_colorspace& full)
{
    printf("Custom data v1 v2 %f %f\n", full.value, full.value2);

    auto transformed = temporary::multiply(full.impl_XYZ_to_linear, temporary::vector_1x3{in.X, in.Y, in.Z});

    out.r = transformed.a[0];
    out.g = transformed.a[1];
    out.b = transformed.a[2];
}

void color_convert(const color::basic_color<fully_custom_colorspace, color::RGB_float_model, color::no_alpha>& in, color::basic_color<color::XYZ_space, color::XYZ_model, color::no_alpha>& out, fully_custom_colorspace& full)
{
    printf("Custom data v1 v2 %f %f\n", full.value, full.value2);

    auto transformed = temporary::multiply(full.impl_linear_to_XYZ, temporary::vector_1x3{in.r, in.g, in.b});

    out.X = transformed.a[0];
    out.Y = transformed.a[1];
    out.Z = transformed.a[2];
}

void tests()
{
    {
        constexpr color::linear_sRGBA_float linear(1, 1, 0.5, 200/255.f);

        constexpr color::sRGBA_uint8 srgb = color::convert<color::sRGBA_uint8>(linear);

        static_assert(color::has_optimised_conversion<color::linear_sRGBA_float, color::sRGBA_uint8>());

        static_assert(approx_equal(srgb.r, 255));
        static_assert(approx_equal(srgb.g, 255));
        static_assert(approx_equal(srgb.b, 188));
        static_assert(approx_equal(srgb.a, 200));
    }

    {
        constexpr retained_type rtype{1,1,1};

        static_assert(color::has_optimised_conversion<retained_type, color::sRGB_uint8>());
    }

    /*{
        fully_custom_colorspace colour_instance;
        colour_instance.value = 53;
        colour_instance.value2 = 999;

        fully_custom_color ccol;
        ccol.r = 1;
        ccol.g = 0;
        ccol.b = 1;

        auto conn = color::make_connector<color::linear_sRGB_float, fully_custom_color>(colour_instance);

        color::linear_sRGB_float converted = conn.convert(ccol);

        printf("Found %f %f %f\n", converted.r, converted.g, converted.b);

        color::linear_sRGB_float converted2 = color::convert<color::linear_sRGB_float>(ccol, colour_instance);

        printf("Found2 %f %f %f\n", converted2.r, converted2.g, converted2.b);
    }*/

    {
        color::sRGB_uint8 val3(255, 127, 80);

        static_assert(color::has_optimised_conversion<color::sRGB_uint8, color::XYZ>());

        color::XYZ test_xyz = color::convert<color::XYZ>(val3);

        static_assert(color::has_optimised_conversion<color::XYZ, color::sRGB_float>());

        color::sRGB_float val4 = color::convert<color::sRGB_float>(test_xyz);

        assert(approx_equal(val4.r, 1));
        assert(approx_equal(val4.g, 0.498039));
        assert(approx_equal(val4.b, 0.313725));
    }

    {
        static_assert(color::has_optimised_conversion<color::sRGBA_uint8, color::sRGBA_float>());

        color::sRGBA_uint8 t1(255, 127, 80, 230);

        color::sRGBA_float t2 = color::convert<color::sRGBA_float>(t1);

        assert(approx_equal(t2.r, 1));
        assert(approx_equal(t2.g, 0.498039));
        assert(approx_equal(t2.b, 0.313726)); ///slightly different results to above due to imprecision
        assert(approx_equal(t2.a, 0.901961));
    }


    {
        color::linear_sRGB_float lin(1, 0, 1);

        static_assert(color::has_optimised_conversion<color::linear_sRGB_float, color::sRGB_float>());

        color::sRGB_float srgb = color::convert<color::sRGB_float>(lin);

        assert(approx_equal(srgb.r, 1));
        assert(approx_equal(srgb.g, 0));
        assert(approx_equal(srgb.b, 1));
    }

    {
        color::linear_sRGB_float lin(0.5, 1, 0.5);

        static_assert(color::has_optimised_conversion<color::linear_sRGB_float, color::sRGB_uint8>());

        color::sRGB_uint8 srgb = color::convert<color::sRGB_uint8>(lin);

        assert(approx_equal(srgb.r, 188));
        assert(approx_equal(srgb.g, 255));
        assert(approx_equal(srgb.b, 188));
    }

    {
        static_assert(std::tuple_size_v<color::RGB_model<color::uint8_value_model, color::uint8_value_model, color::uint8_value_model>> == 3);

        color::RGB_model<color::uint8_value_model, color::uint8_value_model, color::uint8_value_model> model{255, 255, 255};

        auto [x, y, z] = model;

        printf("%i\n", x);
    }
}
#endif // NO_TESTS

int main()
{
    color::sRGB_uint8 val(255, 127, 80);
    color::convert<color::linear_sRGB_float>(val);

    #if 0
    assert(color::has_optimised_conversion(fval, xyz_f));
    assert(color::has_optimised_conversion(custom, fval));

    std::cout << "fval " << fval.r << " " << fval.g << " " << fval.b << std::endl;
    #endif // 0

    #ifndef NO_TESTS
    tests();
    #endif // NO_TESTS

    //color::basic_color<dummy> hello;

    return 0;
}
