#ifndef COLOR_HPP_INCLUDED
#define COLOR_HPP_INCLUDED

#include <type_traits>

#ifdef FIRST_DESIGN
namespace color
{
    template<typename T>
    struct basic_color_space
    {

    };

    template<typename T>
    struct basic_alpha_space
    {
        T alpha = 0;
    };

    class XYZ : basic_color_space<XYZ>
    {
        ///or... use a vector type?
        float X = 0;
        float Y = 0;
        float Z = 0;

        XYZ(float _X, float _Y, float _Z) : X(_X), Y(_Y), Z(_Z)
        {

        }

        XYZ()
        {

        }
    };

    class sRGB_uint8 : basic_color_space<sRGB_uint8>
    {
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;

        sRGB_uint8(uint8_t _r, uint8_t _g, uint8_t _b) : r(_r), g(_g), b(_b)
        {

        }

        sRGB_uint8()
        {

        }
    };

    class sRGB_float : basic_color_space<sRGB_float>
    {
        float r = 0;
        float g = 0;
        float b = 0;

        sRGB_float(float _r, float _g, float _b) : r(_r), g(_g), b(_b)
        {

        }

        sRGB_float()
        {

        }
    };

    class sRGBA_uint8 : basic_color_space<sRGB_uint8>, basic_alpha_space<uint8_t>
    {
        sRGBA_uint8(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a) : r(_r), g(_g), b(_b), a(_a)
        {

        }

        sRGBA_uint8()
        {

        }
    };

    template<typename color_space>
    struct basic_color
    {

    };

    template<typename color_space>
    struct basic_alpha_color
    {

    };
}
#endif // FIRST_DESIGN

#define SECOND_DESIGN
namespace color
{
    template<typename space, typename... tags>
    struct basic_color_space
    {

    };

    template<typename T>
    struct alpha_tag
    {
        T alpha = 0;
    };

    template<typename space, typename... tags>
    inline
    constexpr bool has_alpha(basic_color_space<space, tags...>)
    {
        return (std::is_same_v<tags, alpha_tag> || ...);
    }

    struct XYZ : basic_color_space<XYZ>
    {
        ///or... use a vector type?
        float X = 0;
        float Y = 0;
        float Z = 0;

        XYZ(float _X, float _Y, float _Z) : X(_X), Y(_Y), Z(_Z)
        {

        }

        XYZ()
        {

        }
    };

    struct sRGB_uint8 : basic_color_space<sRGB_uint8>
    {
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;

        sRGB_uint8(uint8_t _r, uint8_t _g, uint8_t _b) : r(_r), g(_g), b(_b)
        {

        }

        sRGB_uint8()
        {

        }
    };

    struct sRGB_float : basic_color_space<sRGB_float>
    {
        float r = 0;
        float g = 0;
        float b = 0;

        sRGB_float(float _r, float _g, float _b) : r(_r), g(_g), b(_b)
        {

        }

        sRGB_float()
        {

        }
    };

    struct sRGBA_uint8 : basic_color_space<sRGBA_uint8, alpha_tag<uint8_t>>, sRGB_uint8, alpha_tag<uint8_t>
    {
        sRGBA_uint8(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a) : sRGB_uint8(_r, _g, _b), alpha_tag<uint8_t>{_a}
        {

        }

        sRGBA_uint8()
        {

        }
    };

    struct sRGBA_float : basic_color_space<sRGBA_float, alpha_tag<uint8_t>>, sRGB_float, alpha_tag<float>
    {
        sRGBA_float(float _r, float _g, float _b, float _a) : sRGB_float(_r, _g, _b), alpha_tag<float>{_a}
        {

        }

        sRGBA_float()
        {

        }
    };
}


#endif // COLOR_HPP_INCLUDED
