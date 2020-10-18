#include "P3.hpp"
#include <assert.h>
#include "common.hpp"

namespace
{
    struct tester
    {
        tester()
        {
            {
                constexpr color::sRGB_float t1(0, 1, 0);

                static_assert(color::has_optimised_conversion<color::sRGB_float, P3_float>());

                std::error_code ec;
                constexpr P3_float t2 = color::convert<P3_float>(t1, ec);

                static_assert(approx_equal(t2.r, 0.458407));
                static_assert(approx_equal(t2.g, 0.985265));
                static_assert(approx_equal(t2.b, 0.29832));
            }

            {
                constexpr color::sRGBA_float t1(0, 1, 0, 1);

                static_assert(color::has_optimised_conversion<color::sRGBA_float, P3A_float>());

                std::error_code ec;
                constexpr P3A_float t2 = color::convert<P3A_float>(t1, ec);

                static_assert(approx_equal(t2.r, 0.458407));
                static_assert(approx_equal(t2.g, 0.985265));
                static_assert(approx_equal(t2.b, 0.29832));
                static_assert(approx_equal(t2.a, 1));
            }

            {
                color::sRGB_float t1(0, 1, 0);

                static_assert(color::has_optimised_conversion<color::sRGB_float, P3_float>());

                std::error_code ec;
                P3_float t2 = color::convert<P3_float>(t1, ec);

                assert(approx_equal(t2.r, 0.458407));
                assert(approx_equal(t2.g, 0.985265));
                assert(approx_equal(t2.b, 0.29832));

                t1 = color::convert<color::sRGB_float>(t2, ec);

                assert(approx_equal(t1.r, 0));
                assert(approx_equal(t1.g, 1));
                assert(approx_equal(t1.b, 0));
            }
        }
    };

    tester test;
}
