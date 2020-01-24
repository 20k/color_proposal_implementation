#ifndef COLOR_HPP_INCLUDED
#define COLOR_HPP_INCLUDED

#include <type_traits>
#include <cmath>

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

#ifdef SECOND_DESIGN
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

    template<typename space_1, typename space_2, typename... tags_1, typename... tags_2>
    basic_color_space
}
#endif // SECOND_DESIGN

#ifdef THIRD_DESIGN
namespace color
{
    template<typename space>
    struct basic_color_space
    {

    };

    struct XYZ_space : basic_color_space<XYZ_space>
    {
        float X = 0;
        float Y = 0;
        float Z = 0;

        XYZ_space(float pX, float pY, float pZ) : X(pX), Y(pY), Z(pZ){}
        XYZ_space(){}
    };

    struct sRGB_uint8_space : basic_color_space<sRGB_uint8_space>
    {
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;

        sRGB_uint8_space(uint8_t pr, uint8_t pg, uint8_t pb) : r(pr), g(pg), b(pb){}
        sRGB_uint8_space(){}
    };

    struct sRGB_float_space : basic_color_space<sRGB_float_space>
    {
        float r = 0;
        float g = 0;
        float b = 0;

        sRGB_float_space(float pr, float pg, float pb) : r(pr), g(pg), b(pb){}
        sRGB_float_space(){}
    };

    template<typename T>
    struct basic_alpha_space
    {
        T alpha = 0;

        basic_alpha_space(const T& palpha) : alpha(palpha){}
        basic_alpha_space(){}
    };

    template<typename... T>
    struct basic_color : T...
    {

    };

    struct sRGB_uint8 : basic_color<sRGB_uint8_space>
    {
        sRGB_uint8(uint8_t pr, uint8_t pg, uint8_t pb) {r = pr; g = pg; b = pb;}
        sRGB_uint8(){}
    };

    struct sRGBA_uint8 : basic_color<sRGB_uint8_space, basic_alpha_space<uint8_t>>
    {
        sRGBA_uint8(uint8_t pr, uint8_t pg, uint8_t pb, uint8_t pa){r = pr; g = pg; b = pb; alpha = pa;}
        sRGBA_uint8(){}
    };

    struct sRGB_float : basic_color<sRGB_float_space>
    {
        sRGB_float(float pr, float pg, float pb){r = pr; g = pg; b = pb;}
        sRGB_float(){}
    };

    struct sRGBA_float : basic_color<sRGB_float_space, basic_alpha_space<float>>
    {
        sRGBA_float(float pr, float pg, float pb, float pa){r = pr; g = pg; b = pb; alpha = pa;}
        sRGBA_float(){}
    };

    struct XYZ_float : basic_color<XYZ_space>
    {
        XYZ_float(float pX, float pY, float pZ){X = pX; Y = pY; Z = pZ;}
        XYZ_float(){}
    };

    /*template<typename space, typename... tags>
    void direct_convert(const basic_color_space<space, tags...>& in)
    {
        static_assert(false, "No conversion");
    }*/

    ///these functions are temporary
    inline
    float lin_single(float in)
    {
        if(in <= 0.04045f)
            return in / 12.92f;
        else
            return std::pow((in + 0.055f) / 1.055f, 2.4f);

    }

    inline
    float gam_single(float in)
    {
        if(in <= 0.0031308f)
            return in * 12.92f;
        else
            return 1.055f * std::pow(in, 1/2.4f) - 0.055f;
    }

    inline
    void direct_convert(const sRGB_float_space& in, XYZ_space& out)
    {
        float fr = in.r;
        float fg = in.g;
        float fb = in.b;

        float lin_r = lin_single(fr);
        float lin_g = lin_single(fg);
        float lin_b = lin_single(fb);

        float X = 0.4124564 * lin_r + 0.3575761 * lin_g + 0.1804375 * lin_b;
        float Y = 0.2126729 * lin_r + 0.7151522 * lin_g + 0.0721750 * lin_b;
        float Z = 0.0193339 * lin_r + 0.1191920 * lin_g + 0.9503041 * lin_b;

        ///todo: constructors
        out.X = X;
        out.Y = Y;
        out.Z = Z;
    }

    inline
    void direct_convert(const sRGB_uint8_space& in, XYZ_space& out)
    {
        sRGB_float f;
        f.r = in.r / 255.f;
        f.g = in.g / 255.f;
        f.b = in.b / 255.f;

        direct_convert(f, out);
    }

    inline
    void direct_convert(const XYZ_space& in, sRGB_float_space& out)
    {
        float lin_r =  3.2404542 * in.X - 1.5371385 * in.Y - 0.4985314 * in.Z;
        float lin_g = -0.9692660 * in.X + 1.8760108 * in.Y + 0.0415560 * in.Z;
        float lin_b =  0.0556434 * in.X - 0.2040259 * in.Y + 1.0572252 * in.Z;

        out.r = gam_single(lin_r);
        out.g = gam_single(lin_g);
        out.b = gam_single(lin_b);
    }

    inline
    void direct_convert(const XYZ_space& in, sRGB_uint8_space& out)
    {
        sRGB_float f;
        direct_convert(in, f);

        out.r = f.r * 255.f;
        out.g = f.g * 255.f;
        out.b = f.b * 255.f;
    }

    template<typename T, typename U, typename = void>
    struct has_direct_conversion_c : std::false_type{};

    template<typename T, typename U>
    struct has_direct_conversion_c<T, U, std::void_t<decltype(direct_convert(std::declval<T>(), std::declval<U&>()))>> : std::true_type{};

    template<typename T, typename U>
    constexpr bool has_optimised_conversion(const T& one, const U& two)
    {
        return has_direct_conversion_c<T, U>::value;
    }

    template<typename from_space_1, typename to_space_2, typename... from_tags_1, typename... to_tags_2>
    inline
    void
    convert(const basic_color<from_space_1, from_tags_1...>& c1, basic_color<to_space_2, to_tags_2...>& c2)
    {
        ///if exists direct conversion use that, otherwise use toXYZ
        if constexpr(has_optimised_conversion(c1, c2))
        {
            return direct_convert(c1, c2);
        }
        else
        {
            XYZ_float intermediate;
            direct_convert(c1, intermediate);
            direct_convert(intermediate, c2);
        }
    }
}
#endif // THIRD_DESIGN

namespace color
{
    struct basic_color_space
    {

    };

    struct basic_color_model
    {

    };

    struct basic_alpha
    {

    };

    ///eg sRGB
    struct static_color_space : basic_color_space
    {

    };

    ///something that might need profiles, runtime information
    struct dynamic_color_space : basic_color_space
    {

    };

    struct sRGB_space : static_color_space
    {

    };

    struct XYZ_space : static_color_space
    {

    };

    template<typename T>
    struct RGB_model : basic_color_model
    {
        T r = 0;
        T g = 0;
        T b = 0;
    };

    struct XYZ_model : basic_color_model
    {
        float X = 0;
        float Y = 0;
        float Z = 0;
    };

    struct RGB_uint8_model : RGB_model<uint8_t>
    {

    };

    struct RGB_float_model : RGB_model<float>
    {

    };

    struct no_alpha : basic_alpha{};

    template<typename T>
    struct simple_alpha : basic_alpha
    {
        T alpha = 0;
    };

    struct float_alpha : simple_alpha<float>{};
    struct uint8_alpha : simple_alpha<uint8_t>{};;

    template<typename cspace, typename cmodel, typename... tags>
    struct basic_color : cmodel, tags...
    {
        using space = cspace;
        using model = cmodel;
    };

    using sRGB_uint8 = basic_color<sRGB_space, RGB_uint8_model, no_alpha>;
    using sRGB_float = basic_color<sRGB_space, RGB_float_model, no_alpha>;
    using XYZ = basic_color<XYZ_space, XYZ_model, no_alpha>;

    using sRGBA_uint8 = basic_color<sRGB_space, RGB_uint8_model, uint8_alpha>;
    using sRGBA_float = basic_color<sRGB_space, RGB_float_model, float_alpha>;

    inline
    float gamma_sRGB_to_linear(float in)
    {
        if(in <= 0.04045f)
            return in / 12.92f;
        else
            return std::pow((in + 0.055f) / 1.055f, 2.4f);
    }

    inline
    float linear_sRGB_to_gamma(float in)
    {
        if(in <= 0.0031308f)
            return in * 12.92f;
        else
            return 1.055f * std::pow(in, 1/2.4f) - 0.055f;
    }

    inline
    void model_convert(const RGB_uint8_model& in, RGB_float_model& out)
    {
        out.r = in.r / 255.f;
        out.g = in.g / 255.f;
        out.b = in.b / 255.f;
    }

    inline
    void model_convert(const RGB_float_model& in, RGB_uint8_model& out)
    {
        ///todo: clamp
        out.r = in.r * 255.f;
        out.g = in.g * 255.f;
        out.b = in.b * 255.f;
    }

    inline
    void color_convert(const sRGB_float& in, XYZ& out)
    {
        float fr = in.r;
        float fg = in.g;
        float fb = in.b;

        float lin_r = gamma_sRGB_to_linear(fr);
        float lin_g = gamma_sRGB_to_linear(fg);
        float lin_b = gamma_sRGB_to_linear(fb);

        float X = 0.4124564 * lin_r + 0.3575761 * lin_g + 0.1804375 * lin_b;
        float Y = 0.2126729 * lin_r + 0.7151522 * lin_g + 0.0721750 * lin_b;
        float Z = 0.0193339 * lin_r + 0.1191920 * lin_g + 0.9503041 * lin_b;

        ///todo: constructors
        out.X = X;
        out.Y = Y;
        out.Z = Z;
    }

    inline
    void color_convert(const XYZ& in, sRGB_float& out)
    {
        float lin_r =  3.2404542 * in.X - 1.5371385 * in.Y - 0.4985314 * in.Z;
        float lin_g = -0.9692660 * in.X + 1.8760108 * in.Y + 0.0415560 * in.Z;
        float lin_b =  0.0556434 * in.X - 0.2040259 * in.Y + 1.0572252 * in.Z;

        out.r = linear_sRGB_to_gamma(lin_r);
        out.g = linear_sRGB_to_gamma(lin_g);
        out.b = linear_sRGB_to_gamma(lin_b);
    }

    inline
    void color_convert(const sRGB_uint8& in, XYZ& out)
    {
        sRGB_float f;
        model_convert(in, f);
        return color_convert(f, out);
    }

    inline
    void color_convert(const XYZ& in, sRGB_uint8& out)
    {
        sRGB_float fout;
        model_convert(out, fout);
        color_convert(in, fout);
        model_convert(fout, out);
    }

    inline
    void alpha_convert(const uint8_alpha& in, float_alpha& out)
    {
        out.alpha = in.alpha / 255.f;
    }

    inline
    void alpha_convert(const float_alpha& in, uint8_alpha& out)
    {
        out.alpha = in.alpha * 255.f;
    }

    template<typename T, typename U, typename = void>
    struct has_direct_conversion_c : std::false_type{};

    template<typename T, typename U>
    struct has_direct_conversion_c<T, U, std::void_t<decltype(color_convert(std::declval<T>(), std::declval<U&>()))>> : std::true_type{};

    template<typename T, typename U>
    constexpr bool has_optimised_conversion(const T& one, const U& two)
    {
        return has_direct_conversion_c<T, U>::value;
    }

    template<typename space_1, typename model_1, typename alpha_1, typename... tags_1, typename space_2, typename model_2, typename alpha_2, typename... tags_2>
    inline
    void convert(const basic_color<space_1, model_1, alpha_1, tags_1...>& in, basic_color<space_2, model_2, alpha_2, tags_2...>& out)
    {
        constexpr bool same_space = std::is_same_v<space_1, space_2>;
        constexpr bool same_model = std::is_same_v<model_1, model_2>;
        constexpr bool same_alpha = std::is_same_v<alpha_1, alpha_2>;
        constexpr bool same_tags = (std::is_same_v<tags_1, tags_2> && ...);

        if constexpr(same_space && same_model && same_tags && same_alpha)
        {
            out = in;
            return;
        }

        if constexpr(same_space && same_tags)
        {
            const model_1& base_model = in;
            model_2& destination_model = out;

            if constexpr(!same_model)
                model_convert(base_model, destination_model);
            else
                destination_model = base_model;

            const alpha_1& base_alpha = in;
            alpha_2& destination_alpha = out;

            if constexpr(!same_alpha)
                alpha_convert(base_alpha, destination_alpha);
            else
                destination_alpha = base_alpha;

            return;
        }
        else
        {
            if constexpr(has_optimised_conversion(in, out))
            {
                color_convert(in, out);
            }
            else
            {
                XYZ intermediate;
                color_convert(in, intermediate);
                color_convert(intermediate, in);
            }
        }
    }
}

#endif // COLOR_HPP_INCLUDED
