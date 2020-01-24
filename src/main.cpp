#include <iostream>

#include "color.hpp"
#include <assert.h>

#if 0
struct custom_color_space : color::basic_color_space<custom_color_space>
{
    float q=0, g=0, b=0, f=0, w=0;
};

struct custom_color : color::basic_color<custom_color_space>
{

};

///this function needs to optionally take a value, aka some sort of user specifiable black box
///eg in the case of SFML, we could pass in a renderwindow and make decisions based on gamma handling
///and have a custom colour type which is always correct regardless of the environment
void direct_convert(const custom_color& in, color::sRGB_float& test)
{

}
#endif // 0

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

    color::sRGB_uint8 val;
    val.r = 255;
    val.g = 127;
    val.b = 80;

    color::sRGB_float val2;

    ///performs a model conversion
    color::convert(val, val2);


    color::sRGB_uint8 val3;
    val3.r = 255;
    val3.g = 127;
    val3.b = 80;

    color::XYZ test_xyz;

    color::convert(val3, test_xyz);

    //color::basic_color<dummy> hello;

    return 0;
}
