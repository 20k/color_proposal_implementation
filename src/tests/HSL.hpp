#pragma once

#include <color.hpp>
#include <math.h>

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
inline
void color_convert(const color::basic_color<HSL_space, hsl_model, hsl_alpha>& in, color::basic_color<color::generic_RGB_space<space_1, gamma_1>, model_1, alpha_1>& out)
{
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