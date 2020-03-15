#include "slow_sRGB.hpp"
#include <assert.h>
#include "common.hpp"

namespace 
{
    struct tester
    {
        tester()
        {
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
        }
    };

    tester test;
}