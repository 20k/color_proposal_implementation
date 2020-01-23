#include <iostream>

#include "color.hpp"
#include <assert.h>

struct custom_color_space : color::basic_color_space<custom_color_space>
{

};

struct custom_color : color::basic_color<custom_color_space>
{

};

void direct_convert(const custom_color& in, color::sRGB_float& test)
{

}

int main()
{
    color::sRGB_uint8 val(255, 127, 80);
    color::sRGB_float fval;
    color::XYZ_float xyz_f(0, 0, 0);
    custom_color custom;

    color::convert(val, fval);

    assert(color::has_optimised_conversion(fval, xyz_f));
    assert(color::has_optimised_conversion(custom, fval));

    std::cout << "fval " << fval.r << " " << fval.g << " " << fval.b << std::endl;

    return 0;
}
