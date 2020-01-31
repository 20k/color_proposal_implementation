#include <iostream>

#include "color.hpp"
#include <assert.h>
#include <algorithm>

struct HSL_float_value_model
{
    using type = float;

    static inline constexpr float base = 0;
    static inline constexpr float scale = 360;

    static inline constexpr float min = -FLT_MAX;
    static inline constexpr float max = FLT_MAX;
};

template<typename V1, typename V2, typename V3>
struct HSL_model : color::basic_color_model
{
    typename V1::type h = typename V1::type();
    typename V2::type s = typename V2::type();
    typename V3::type l = typename V3::type();

    using H_value = V1;
    using S_value = V2;
    using L_value = V3;

    constexpr HSL_model(typename V1::type _h, typename V2::type _s, typename V3::type _l) : h(_h), s(_s), l(_l){}
    constexpr HSL_model(){}
};

struct HSL_space{};

using HSL_float_model = HSL_model<HSL_float_value_model, color::normalised_float_value_model, color::normalised_float_value_model>;

struct HSL_360 : color::basic_color<HSL_space, HSL_float_model, color::no_alpha>
{
    constexpr HSL_360(float hue_angle_360, float _S, float _L)
    {
        h = hue_angle_360;
        s = _S;
        l = _L;
    }

    constexpr HSL_360()
    {
        h = 0;
        s = 0;
        l = 0;
    }
};

template<typename hsl_model, typename hsl_alpha, typename space_1, typename model_1, typename gamma_1, typename alpha_1>
void color_convert(const color::basic_color<HSL_space, hsl_model, hsl_alpha>& in, color::basic_color<color::generic_RGB_space<space_1, gamma_1>, model_1, alpha_1>& out)
{
    ///convert to sRGB
    ///farm off to ::convert

    using namespace color;

    ///all [0 -> 1]
    float nH = 0;
    float nS = 0;
    float nL = 0;

    model_convert_member<typename hsl_model::H_value, normalised_float_value_model>(in.h, nH);
    model_convert_member<typename hsl_model::S_value, normalised_float_value_model>(in.s, nS);
    model_convert_member<typename hsl_model::L_value, normalised_float_value_model>(in.l, nL);

    float min_l = std::min(nL, 1 - nL);

    float a = nS * min_l;

    auto hsl_fn = [](int n, float a, float nH, float nL)
    {
        float k = fmod((n + (360 / 30) * nH), 12.f);
        float min_k = k - 3;

        min_k = std::min(min_k, 9 - k);
        min_k = std::min(min_k, 1.f);
        min_k = std::max(min_k,-1.f);

        return nL - a * min_k;
    };

    basic_color<sRGB_space, RGB_float_model, hsl_alpha> intermediate;
    intermediate.r = hsl_fn(0, a, nH, nL);
    intermediate.g = hsl_fn(8, a, nH, nL);
    intermediate.b = hsl_fn(4, a, nH, nL);

    alpha_convert(in, intermediate);

    color_convert(intermediate, out);
}

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

    using gamma = color::transfer_function::default_parameterisation;
};

using adobe_space = color::generic_RGB_space<adobe_RGB_98_parameters, adobe_RGB_98_transfer_parameters>;
//using adobe_float = color::basic_color<adobe_space, color::RGB_float_model>;

struct adobe_float : color::basic_color<adobe_space, color::RGB_float_model, color::no_alpha>
{
    constexpr adobe_float(float _r, float _g, float _b) {r = _r; g = _g; b = _b;}
    constexpr adobe_float(){}
};

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
constexpr
void color_convert(const color::basic_color<color::XYZ_space, color::XYZ_model, A1>& in, color::basic_color<slow_sRGB_space, slow_sRGB_model, A2>& out)
{
    auto transformed = temporary::multiply(slow_sRGB_space::impl_XYZ_to_linear, (temporary::vector_1x3){in.X, in.Y, in.Z});

    out.r = transformed.a[0];
    out.g = transformed.a[1];
    out.b = transformed.a[2];

    color::alpha_convert(in, out);
}

template<typename A1, typename A2>
constexpr
void color_convert(const color::basic_color<slow_sRGB_space, slow_sRGB_model, A1>& in, color::basic_color<color::XYZ_space, color::XYZ_model, A2>& out)
{
    auto transformed = temporary::multiply(slow_sRGB_space::impl_linear_to_XYZ, (temporary::vector_1x3){in.r, in.g, in.b});

    out.X = transformed.a[0];
    out.Y = transformed.a[1];
    out.Z = transformed.a[2];

    color::alpha_convert(in, out);
}

void color_convert(const color::basic_color<color::XYZ_space, color::XYZ_model, color::no_alpha>& in, color::basic_color<fully_custom_colorspace, color::RGB_float_model, color::no_alpha>& out, fully_custom_colorspace& full)
{
    printf("Custom data v1 v2 %f %f\n", full.value, full.value2);

    auto transformed = temporary::multiply(full.impl_XYZ_to_linear, (temporary::vector_1x3){in.X, in.Y, in.Z});

    out.r = transformed.a[0];
    out.g = transformed.a[1];
    out.b = transformed.a[2];
}

void color_convert(const color::basic_color<fully_custom_colorspace, color::RGB_float_model, color::no_alpha>& in, color::basic_color<color::XYZ_space, color::XYZ_model, color::no_alpha>& out, fully_custom_colorspace& full)
{
    printf("Custom data v1 v2 %f %f\n", full.value, full.value2);

    auto transformed = temporary::multiply(full.impl_linear_to_XYZ, (temporary::vector_1x3){in.r, in.g, in.b});

    out.X = transformed.a[0];
    out.Y = transformed.a[1];
    out.Z = transformed.a[2];
}

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

        static_assert(color::has_optimised_conversion<color::sRGB_float, P3_float>());

        constexpr P3_float t2 = color::convert<P3_float>(t1);

        static_assert(approx_equal(t2.r, 0.458407));
        static_assert(approx_equal(t2.g, 0.985265));
        static_assert(approx_equal(t2.b, 0.29832));
    }

    {
        constexpr color::sRGBA_float t1(0, 1, 0, 1);

        static_assert(color::has_optimised_conversion<color::sRGBA_float, P3A_float>());

        constexpr P3A_float t2 = color::convert<P3A_float>(t1);

        static_assert(approx_equal(t2.r, 0.458407));
        static_assert(approx_equal(t2.g, 0.985265));
        static_assert(approx_equal(t2.b, 0.29832));
        static_assert(approx_equal(t2.a, 1));
    }

    {
        constexpr HSL_360 hsl(195, 1, 0.5);

        color::sRGB_uint8 srgb = color::convert<color::sRGB_uint8>(hsl);

        assert(srgb.r == 0);
        assert(srgb.g == 191);
        assert(srgb.b == 255);
    }

    {
        constexpr weirdo_linear_space i_dislike_type_safety(255, 255, 128);

        static_assert(color::has_optimised_conversion<color::sRGB_float, weirdo_linear_space>());
        static_assert(color::has_optimised_conversion<P3_float, weirdo_linear_space>());
        static_assert(color::has_optimised_conversion<color::linear_sRGB_float, weirdo_linear_space>());
        static_assert(color::has_optimised_conversion<color::sRGB_uint8, weirdo_linear_space>());

        constexpr color::linear_sRGB_float linear = color::convert<color::linear_sRGB_float>(i_dislike_type_safety);

        static_assert(approx_equal(linear.r, 1.f));
        static_assert(approx_equal(linear.g, 1.f));
        static_assert(approx_equal(linear.b, (128/255.f)));

        constexpr color::sRGB_uint8 srgb = color::convert<color::sRGB_uint8>(i_dislike_type_safety);

        static_assert(approx_equal(srgb.r, 255));
        static_assert(approx_equal(srgb.g, 255));
        static_assert(approx_equal(srgb.b, 188));
    }

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
        constexpr color::linear_sRGBA_float linear(1, 1, 0.5, 200/255.f);

        static_assert(!color::has_optimised_conversion<slow_linear_sRGBA_color, color::linear_sRGBA_float>());

        constexpr slow_linear_sRGBA_color cv = color::convert<slow_linear_sRGBA_color>(linear);

        static_assert(approx_equal(cv.r, 1));
        static_assert(approx_equal(cv.g, 1));
        static_assert(approx_equal(cv.b, 0.5));
        static_assert(approx_equal(cv.a, 200/255.f));
    }

    {
        constexpr color::sRGBA_uint8 srgb(160, 170, 100, 50);

        static_assert(!color::has_optimised_conversion<slow_linear_sRGBA_color, color::sRGBA_uint8>());

        constexpr slow_linear_sRGBA_color cv = color::convert<slow_linear_sRGBA_color>(srgb);

        constexpr color::linear_sRGBA_float cv2 = color::convert<color::linear_sRGBA_float>(srgb);

        static_assert(approx_equal(cv.r, cv2.r));
        static_assert(approx_equal(cv.g, cv2.g));
        static_assert(approx_equal(cv.b, cv2.b));
        static_assert(approx_equal(cv.a, cv2.a));
    }

    {
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
    }

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
        color::sRGB_float t1(0, 1, 0);

        static_assert(color::has_optimised_conversion<color::sRGB_float, P3_float>());

        P3_float t2 = color::convert<P3_float>(t1);

        assert(approx_equal(t2.r, 0.458407));
        assert(approx_equal(t2.g, 0.985265));
        assert(approx_equal(t2.b, 0.29832));

        t1 = color::convert<color::sRGB_float>(t2);

        assert(approx_equal(t1.r, 0));
        assert(approx_equal(t1.g, 1));
        assert(approx_equal(t1.b, 0));
    }

    {
        color::sRGB_float t1(0, 1, 0);

        static_assert(color::has_optimised_conversion<color::sRGB_float, adobe_float>());

        adobe_float t2 = color::convert<adobe_float>(t1);

        assert(approx_equal(t2.r, 0.564978));
        assert(approx_equal(t2.g, 1));
        assert(approx_equal(t2.b, 0.234443));
    }

    {
        adobe_float t1(1, 0, 1);

        static_assert(color::has_optimised_conversion<adobe_float, color::XYZ>());

        color::XYZ t2 = color::convert<color::XYZ>(t1);
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
