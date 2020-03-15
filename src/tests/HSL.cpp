#include "HSL.hpp"
#include <assert.h>

namespace 
{
    struct tester
    {
        tester()
        {
            {
                constexpr HSL_360 hsl(195, 1, 0.5);

                color::sRGB_uint8 srgb = color::convert<color::sRGB_uint8>(hsl);

                assert(srgb.r == 0);
                assert(srgb.g == 191);
                assert(srgb.b == 255);
            }
        }
    };

    tester test;
}