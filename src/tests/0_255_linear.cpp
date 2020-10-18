#include "0_255_linear.hpp"
#include <assert.h>
#include "common.hpp"

namespace
{
    struct tester
    {
        tester()
        {
            {
                constexpr weirdo_linear_space i_dislike_type_safety(255, 255, 128);

                static_assert(color::has_optimised_conversion<color::sRGB_float, weirdo_linear_space>());
                static_assert(color::has_optimised_conversion<color::linear_sRGB_float, weirdo_linear_space>());
                static_assert(color::has_optimised_conversion<color::sRGB_uint8, weirdo_linear_space>());

                std::error_code ec;
                constexpr color::linear_sRGB_float linear = color::convert<color::linear_sRGB_float>(i_dislike_type_safety, ec);

                static_assert(approx_equal(linear.r, 1.f));
                static_assert(approx_equal(linear.g, 1.f));
                static_assert(approx_equal(linear.b, (128/255.f)));

                std::error_code ec2;
                constexpr color::sRGB_uint8 srgb = color::convert<color::sRGB_uint8>(i_dislike_type_safety, ec2);

                static_assert(approx_equal(srgb.r, 255));
                static_assert(approx_equal(srgb.g, 255));
                static_assert(approx_equal(srgb.b, 188));
            }
        }
    };

    tester test;
}
