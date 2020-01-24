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

namespace temporary
{
    struct matrix_3x3
    {
        float a[3][3] = {0};

        constexpr float determinant()
        {
            float a11=0, a12=0, a13=0, a21=0, a22=0, a23=0, a31=0, a32=0, a33=0;

            a11 = a[0][0];
            a12 = a[0][1];
            a13 = a[0][2];

            a21 = a[1][0];
            a22 = a[1][1];
            a23 = a[1][2];

            a31 = a[2][0];
            a32 = a[2][1];
            a33 = a[2][2];

            return a11*a22*a33 + a21*a32*a13 + a31*a12*a23 - a11*a32*a23 - a31*a22*a13 - a21*a12*a33;
        }

        constexpr matrix_3x3 invert()
        {
            float id = 1.f/determinant();

            float a11=0, a12=0, a13=0, a21=0, a22=0, a23=0, a31=0, a32=0, a33=0;

            a11 = a[0][0];
            a12 = a[0][1];
            a13 = a[0][2];

            a21 = a[1][0];
            a22 = a[1][1];
            a23 = a[1][2];

            a31 = a[2][0];
            a32 = a[2][1];
            a33 = a[2][2];

            float t0 = a22 * a33 - a23 * a32;
            float t1 = a13 * a32 - a12 * a33;
            float t2 = a12 * a23 - a13 * a22;

            float m0 = a23 * a31 - a21 * a33;
            float m1 = a11 * a33 - a13 * a31;
            float m2 = a13 * a21 - a11 * a23;

            float b0 = a21 * a32 - a22 * a31;
            float b1 = a12 * a31 - a11 * a32;
            float b2 = a11 * a22 - a12 * a21;

            t0 *= id;
            t1 *= id;
            t2 *= id;

            m0 *= id;
            m1 *= id;
            m2 *= id;

            b0 *= id;
            b1 *= id;
            b2 *= id;

            return {{{t0, t1, t2}, {m0, m1, m2}, {b0, b1, b2}}};
        }
    };

    constexpr matrix_3x3 multiply(const matrix_3x3& one, const matrix_3x3& two)
    {
        matrix_3x3 ret;

        for(int i=0; i < 3; i++)
        {
            for(int j=0; j < 3; j++)
            {
                float sum = 0;

                for(int k=0; k < 3; k++)
                {
                    sum += one.a[i][k] * two.a[k][j];
                }

                ret.a[i][j] = sum;
            }
        }

        return ret;
    }

    struct vector_1x3
    {
        float a[3] = {0};
    };

    constexpr vector_1x3 multiply(const matrix_3x3& one, const vector_1x3& vec)
    {
        vector_1x3 ret;
        ret.a[0] = one.a[0][0] * vec.a[0] + one.a[0][1] * vec.a[1] + one.a[0][2] * vec.a[2];
        ret.a[1] = one.a[1][0] * vec.a[0] + one.a[1][1] * vec.a[1] + one.a[1][2] * vec.a[2];
        ret.a[2] = one.a[2][0] * vec.a[0] + one.a[2][1] * vec.a[1] + one.a[2][2] * vec.a[2];

        return ret;
    }
}

///problems with current design
///arbitrary rgb -> xyz -> arbitrary rgb could be expressed as 1 matrix, but we have no way to express that. This is why folks uses connector objects
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

    struct chromaticity
    {
        float x = 0;
        float y = 0;
    };

    namespace illuminant
    {
        namespace CIE1931
        {
            static constexpr chromaticity D65{0.31271, 0.32902};
        }

        namespace CIE1964
        {
            static constexpr chromaticity D65{0.31382, 0.33100};
        }
    }

    struct basic_generic_RGB_tag
    {

    };

    constexpr temporary::matrix_3x3 get_XYZ_to_linear_RGB(chromaticity R, chromaticity G, chromaticity B, chromaticity W)
    {
        float xr=0, yr=0, zr=0, xg=0, yg=0, zg=0, xb=0, yb=0, zb=0;
        float xw=0, yw=0, zw=0;
        float rx=0, ry=0, rz=0, gx=0, gy=0, gz=0, bx=0, by=0, bz=0;
        float rw=0, gw=0, bw=0;

        xr = R.x;  yr = R.y;  zr = 1 - (xr + yr);
        xg = G.x;  yg = G.y;  zg = 1 - (xg + yg);
        xb = B.x;  yb = B.y;  zb = 1 - (xb + yb);

        xw = W.x;  yw = W.y;  zw = 1 - (xw + yw);

        rx = (yg * zb) - (yb * zg);  ry = (xb * zg) - (xg * zb);  rz = (xg * yb) - (xb * yg);
        gx = (yb * zr) - (yr * zb);  gy = (xr * zb) - (xb * zr);  gz = (xb * yr) - (xr * yb);
        bx = (yr * zg) - (yg * zr);  by = (xg * zr) - (xr * zg);  bz = (xr * yg) - (xg * yr);

        rw = ((rx * xw) + (ry * yw) + (rz * zw)) / yw;
        gw = ((gx * xw) + (gy * yw) + (gz * zw)) / yw;
        bw = ((bx * xw) + (by * yw) + (bz * zw)) / yw;

        rx = rx / rw;  ry = ry / rw;  rz = rz / rw;
        gx = gx / gw;  gy = gy / gw;  gz = gz / gw;
        bx = bx / bw;  by = by / bw;  bz = bz / bw;

        return {{{rx, ry, rz}, {gx, gy, gz}, {bx, by, bz}}};
    }

    constexpr temporary::matrix_3x3 get_linear_RGB_to_XYZ(chromaticity R, chromaticity G, chromaticity B, chromaticity W)
    {
        return get_XYZ_to_linear_RGB(R, G, B, W).invert();
    }

    struct sRGB_parameters
    {
        static constexpr chromaticity R{0.64, 0.33};
        static constexpr chromaticity G{0.30, 0.60};
        static constexpr chromaticity B{0.15, 0.06};
        static constexpr chromaticity W = illuminant::CIE1931::D65;

        static constexpr temporary::matrix_3x3 linear_to_XYZ = get_linear_RGB_to_XYZ(R, G, B, W);
    };

    ///eg sRGB
    struct static_color_space : basic_color_space
    {

    };

    ///something that might need profiles, runtime information
    template<typename T>
    struct dynamic_color_space : basic_color_space
    {
        T& dynamic_data;
    };

    template<typename T>
    struct generic_RGB_space : T, basic_generic_RGB_tag, static_color_space
    {

    };

    struct sRGB_space : generic_RGB_space<sRGB_parameters>
    {
        using RGB_parameters = sRGB_parameters;
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

    template<typename cspace, typename cmodel, typename calpha, typename... tags>
    struct basic_color : cmodel, calpha, tags...
    {
        using space_type = cspace;
        using model_type = cmodel;
        using alpha_type = calpha;
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

    template<typename C1, typename C2>
    inline
    constexpr void color_convert(const generic_RGB_space<C1>& one, const generic_RGB_space<C2>& two)
    {
        /*using params = sRGB_float::space_type::RGB_parameters;

        constexpr temporary::matrix_3x3 mat = get_linear_RGB_to_XYZ(params::R, params::G, params::B, params::W);

        auto vec = temporary::multiply(mat, (temporary::vector_1x3){lin_r, lin_g, lin_b});

        float X = vec.a[0];
        float Y = vec.a[1];
        float Z = vec.a[2];*/
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

        float X = 0.4124564f * lin_r + 0.3575761f * lin_g + 0.1804375f * lin_b;
        float Y = 0.2126729f * lin_r + 0.7151522f * lin_g + 0.0721750f * lin_b;
        float Z = 0.0193339f * lin_r + 0.1191920f * lin_g + 0.9503041f * lin_b;

        ///todo: constructors
        out.X = X;
        out.Y = Y;
        out.Z = Z;
    }

    inline
    void color_convert(const XYZ& in, sRGB_float& out)
    {
        float lin_r =  3.2404542f * in.X - 1.5371385f * in.Y - 0.4985314f * in.Z;
        float lin_g = -0.9692660f * in.X + 1.8760108f * in.Y + 0.0415560f * in.Z;
        float lin_b =  0.0556434f * in.X - 0.2040259f * in.Y + 1.0572252f * in.Z;

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
