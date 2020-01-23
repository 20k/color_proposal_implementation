#include <iostream>

#include "color.hpp"

int main()
{
    color::sRGB_uint8 val(255, 127, 80);
    color::sRGB_float fval;

    color::convert(val, fval);

    std::cout << "fval " << fval.r << " " << fval.g << " " << fval.b << std::endl;

    return 0;
}
