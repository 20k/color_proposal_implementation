#include "adobe_RGB_98.hpp"
#include <assert.h>
#include "common.hpp"

namespace 
{
    struct tester
    {
        tester()
        {
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
        }
    };

    tester test;
}