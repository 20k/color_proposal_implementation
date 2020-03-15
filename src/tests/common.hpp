#pragma once

constexpr inline bool approx_equal(float v1, float v2)
{
    if(v1 < v2)
        return (v2 - v1) < 0.000001f;
    else
        return (v1 - v2) < 0.000001f;
}
