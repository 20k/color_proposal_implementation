/*
* color.hpp
* 2020-01-19 18:31:17
* J. Guy Davidson
*/

#ifndef COLOR_HPP_INCLUDED
#define COLOR_HPP_INCLUDED

#include <type_traits>
#include <cmath>
#include <cfloat>
#include <tuple>
#include <array>

namespace temporary
{
    struct matrix_3x3
    {
        float a[3][3] = {0};

        constexpr float determinant() const
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

        constexpr matrix_3x3 invert() const
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

    template<typename T1, typename T2>
    inline constexpr
    void model_convert_member(const typename T1::type& in, typename T2::type& out)
    {
        auto intermediate = ((in - T1::base) / (T1::scale)) * (T2::scale) + T2::base;

        if(std::is_integral_v<typename T2::type>)
            intermediate = round(intermediate);

        if(intermediate < T2::min)
            out = T2::min;
        else if(intermediate > T2::max)
            out = T2::max;
        else
            out = intermediate;
    }

    ///min and max are clamp values
    ///to convert to this models space, a normalised float value is multiplied by scale, and then base is added

    struct normalised_float_value_model
    {
        using type = float;

        static inline constexpr float base = 0;
        static inline constexpr float scale = 1;

        static inline constexpr float min = 0;
        static inline constexpr float max = 1;
    };

    struct uint8_value_model
    {
        using type = uint8_t;

        static inline constexpr float base = 0;
        static inline constexpr float scale = 255;

        static inline constexpr float min = 0;
        static inline constexpr float max = 255;
    };

    struct unnormalised_float_value_model
    {
        using type = float;

        static inline constexpr float base = 0;
        static inline constexpr float scale = 1;

        static inline constexpr float min = -FLT_MAX;
        static inline constexpr float max = FLT_MAX;
    };

    template<typename T>
    struct concrete_value_model
    {
        typename T::type v = 0;
        using type = T;
    };

    namespace transfer_function
    {
        struct default_parameterisation
        {
            template<typename T, typename U>
            static inline constexpr
            concrete_value_model<normalised_float_value_model> gamma_to_linear(concrete_value_model<T> real_component, const U& in)
            {
                float component = 0;
                model_convert_member<T, normalised_float_value_model>(real_component.v, component);

                if(component <= in.transfer_bdelta)
                    return {component / in.transfer_delta};
                else
                    return {std::pow((component + in.transfer_alpha - 1) / in.transfer_alpha, in.transfer_gamma)};
            }

            template<typename U, typename value_model, typename model_in>
            static inline constexpr
            concrete_value_model<value_model> linear_to_gamma(concrete_value_model<model_in> component, const U& in, value_model tag)
            {
                typename model_in::type my_val = 0;

                if(component.v <= in.transfer_beta)
                    my_val = component.v * in.transfer_delta;
                else
                    my_val = in.transfer_alpha * std::pow(component.v, 1/in.transfer_gamma) - (in.transfer_alpha - 1);

                concrete_value_model<value_model> ret = concrete_value_model<value_model>();
                model_convert_member<model_in, value_model>(my_val, ret.v);
                return ret;
            }
        };

        struct none
        {
            template<typename value_model, typename U>
            static inline constexpr
            concrete_value_model<normalised_float_value_model> gamma_to_linear(concrete_value_model<value_model> real_component, const U& in)
            {
                float component = 0;
                model_convert_member<value_model, normalised_float_value_model>(real_component.v, component);
                return {component};
            }

            template<typename value_model, typename U, typename model_in>
            static inline constexpr
            concrete_value_model<value_model> linear_to_gamma(concrete_value_model<model_in> real_component, const U& in, value_model tag)
            {
                concrete_value_model<value_model> component = concrete_value_model<value_model>();
                model_convert_member<model_in, value_model>(real_component.v, component.v);
                return component;
            }
        };
    }

    struct sRGB_parameters
    {
        static constexpr chromaticity R{0.64, 0.33};
        static constexpr chromaticity G{0.30, 0.60};
        static constexpr chromaticity B{0.15, 0.06};
        static constexpr chromaticity W = illuminant::CIE1931::D65;

        ///TODO: need to use rounded sRGB parameters from the spec rather than calculating them
        static constexpr temporary::matrix_3x3 linear_to_XYZ = get_linear_RGB_to_XYZ(R, G, B, W);
        static constexpr temporary::matrix_3x3 XYZ_to_linear = linear_to_XYZ.invert();
    };

    struct sRGB_transfer_parameters
    {
        static constexpr float transfer_alpha = 1.055;
        static constexpr float transfer_beta = 0.0031308;
        static constexpr float transfer_gamma = 12/5.f;
        static constexpr float transfer_delta = 12.92;
        static constexpr float transfer_bdelta = 0.04045;

        using gamma = transfer_function::default_parameterisation;
    };

    struct empty_transfer_parameters
    {
        using gamma = transfer_function::none;
    };

    template<typename T, typename U>
    struct generic_RGB_space : basic_color_space
    {
        using RGB_parameters = T;
        using transfer_function_parameters = U;
    };

    using sRGB_space = generic_RGB_space<sRGB_parameters, sRGB_transfer_parameters>;
    ///'linear color' is really linear sRGB specifically
    using linear_sRGB_space = generic_RGB_space<sRGB_parameters, empty_transfer_parameters>;

    struct XYZ_space : basic_color_space
    {

    };

    template<typename VA>
    struct alpha_model
    {
        typename VA::type a = typename VA::type();

        using A_value = VA;

        constexpr alpha_model(typename VA::type _a){a = _a;}
        constexpr alpha_model(){}
    };

    template<>
    struct alpha_model<void>
    {

    };

    using float_alpha = alpha_model<normalised_float_value_model>;
    using uint8_alpha = alpha_model<uint8_value_model>;
    using no_alpha = alpha_model<void>;

    /// SO THINK I FINALLY HAVE A RESOLUTION
    /// For RGB models, make them a tuple
    /// Implement mod.get(idx), or std::get<N>(type)
    /// Then, in composite model, implement mod.get(idx), where idx > first base = second base. Generic model combiner
    /// Then we have introspection over value models if we get value models out, or use traits and specialise (?). Kind of hate traits though
    /// So, once we can iterate over a composite model, we can pull out the underlying properties
    /// std::is_base_of alpha model, for things which are composites of RGB_model and alpha_model
    /// Auto generate tables for that lot
    /// Then any type which implements the type introspection can have tables autogenerated
    /// Eg if your type provides num_color() and num_alpha(), and color(idx) and alpha(idx), you can do it
    /// Or more realistically, type tags, some way to query the type of a member, a .get(), and a .size()
    template<typename V1, typename V2, typename V3>
    struct RGB_model : basic_color_model
    {
        typename V1::type r = typename V1::type();
        typename V2::type g = typename V2::type();
        typename V3::type b = typename V3::type();

        using R_value = V1;
        using G_value = V2;
        using B_value = V3;

        constexpr RGB_model(typename V1::type _r, typename V2::type _g, typename V3::type _b) : r(_r), g(_g), b(_b){}
        constexpr RGB_model(){}
    };

    /*template<typename V1, typename V2, typename V3, typename V4>
    struct RGBA_model : basic_color_model
    {
        typename V1::type r = typename V1::type();
        typename V2::type g = typename V2::type();
        typename V3::type b = typename V3::type();
        typename V4::type a = typename V4::type();

        using R_value = V1;
        using G_value = V2;
        using B_value = V3;
        using A_value = V4;

        constexpr RGBA_model(typename V1::type _r, typename V2::type _g, typename V3::type _b, typename V4::type _a) : r(_r), g(_g), b(_b), a(_a){}
        constexpr RGBA_model(){}
    };*/

    struct XYZ_model : basic_color_model
    {
        float X = 0;
        float Y = 0;
        float Z = 0;
    };

    using RGB_uint8_model = RGB_model<uint8_value_model, uint8_value_model, uint8_value_model>;
    using RGB_float_model = RGB_model<normalised_float_value_model, normalised_float_value_model, normalised_float_value_model>;
    //using RGBA_uint8_model = RGBA_model<uint8_value_model, uint8_value_model, uint8_value_model, uint8_value_model>;
    //using RGBA_float_model = RGBA_model<normalised_float_value_model, normalised_float_value_model, normalised_float_value_model, normalised_float_value_model>;

    template<typename cspace, typename cmodel, typename calpha>
    struct basic_color : cmodel, calpha
    {
        using space_type = cspace;
        using model_type = cmodel;
        using alpha_type = calpha;
        using cmodel::cmodel;
        using calpha::calpha;
    };

    template<typename T, typename U>
    struct composite_model : T, U
    {

    };

    struct sRGB_uint8 : basic_color<sRGB_space, RGB_uint8_model, no_alpha>
    {
        constexpr sRGB_uint8(uint8_t _r, uint8_t _g, uint8_t _b){r = _r; g = _g; b = _b;}
        constexpr sRGB_uint8(){}
    };

    struct sRGB_float : basic_color<sRGB_space, RGB_float_model, no_alpha>
    {
        constexpr sRGB_float(float _r, float _g, float _b){r = _r; g = _g; b = _b;}
        constexpr sRGB_float(){}
    };

    struct sRGBA_uint8 : basic_color<sRGB_space, RGB_uint8_model, uint8_alpha>
    {
        constexpr sRGBA_uint8(uint8_t _r, uint8_t _g, uint8_t _b, uint8_t _a){r = _r; g = _g; b = _b; a = _a;}
        constexpr sRGBA_uint8(){}
    };

    struct sRGBA_float : basic_color<sRGB_space, RGB_float_model, float_alpha>
    {
        constexpr sRGBA_float(float _r, float _g, float _b, float _a){r = _r; g = _g; b = _b; a = _a;}
        constexpr sRGBA_float(){}
    };

    struct linear_sRGB_float : basic_color<linear_sRGB_space, RGB_float_model, no_alpha>
    {
        constexpr linear_sRGB_float(float _r, float _g, float _b){r = _r; g = _g; b = _b;}
        constexpr linear_sRGB_float(){}
    };

    struct linear_sRGBA_float : basic_color<linear_sRGB_space, RGB_float_model, float_alpha>
    {
        constexpr linear_sRGBA_float(float _r, float _g, float _b, float _a){r = _r; g = _g; b = _b; a = _a;}
        constexpr linear_sRGBA_float(){}
    };

    struct XYZ : basic_color<XYZ_space, XYZ_model, no_alpha>
    {
        constexpr XYZ(float _X, float _Y, float _Z){X = _X; Y = _Y; Z = _Z;}
        constexpr XYZ(){}
    };

    template<typename T1, typename T2>
    inline
    constexpr void alpha_convert(const alpha_model<T1>& in, alpha_model<T2>& out)
    {
        static_assert(std::is_same_v<T1, void> == std::is_same_v<T2, void>, "Cannot drop or gain an alpha component");

        if constexpr(std::is_same_v<T1, void>)
            return;
        else
            model_convert_member<T1, T2>(in.a, out.a);
    }

    template<typename T1, typename U1, typename V1,
             typename T2, typename U2, typename V2>
    inline
    constexpr void model_convert(const RGB_model<T1, U1, V1>& in, RGB_model<T2, U2, V2>& out)
    {
        model_convert_member<T1, T2>(in.r, out.r);
        model_convert_member<U1, U2>(in.g, out.g);
        model_convert_member<V1, V2>(in.b, out.b);
    }

    /*using sRGB_uint8 = basic_color<sRGB_space, RGB_uint8_model>;
    using sRGB_float = basic_color<sRGB_space, RGB_float_model>;
    using sRGBA_uint8 = basic_color<sRGB_space, RGBA_uint8_model>;
    using sRGBA_float = basic_color<sRGB_space, RGBA_float_model>;
    using linear_sRGB_float = basic_color<linear_sRGB_space, RGB_float_model>;
    using linear_sRGBA_float = basic_color<linear_sRGB_space, RGBA_float_model>;
    using XYZ = basic_color<XYZ_space, XYZ_model>;*/

    /*
    template<typename T1, typename U1, typename V1, typename W1,
             typename T2, typename U2, typename V2, typename W2>
    inline
    constexpr void model_convert(const RGBA_model<T1, U1, V1, W1>& in, RGBA_model<T2, U2, V2, W2>& out)
    {
        model_convert_member<T1, T2>(in.r, out.r);
        model_convert_member<U1, U2>(in.g, out.g);
        model_convert_member<V1, V2>(in.b, out.b);
        model_convert_member<W1, W2>(in.a, out.a);
    }*/

    ///direct conversion between two arbitrary RGB space
    template<typename space_1, typename model_1, typename gamma_1, typename alpha_1, typename space_2, typename model_2, typename gamma_2, typename alpha_2>
    inline
    constexpr void color_convert(const basic_color<generic_RGB_space<space_1, gamma_1>, model_1, alpha_1>& in, basic_color<generic_RGB_space<space_2, gamma_2>, model_2, alpha_2>& out)
    {
        ///different models and alpha
        if constexpr(std::is_same_v<space_1, space_2> && std::is_same_v<gamma_1, gamma_2>)
        {
            model_convert(in, out);
        }
        else
        {
            concrete_value_model<typename model_1::R_value> rv{in.r};
            concrete_value_model<typename model_1::G_value> gv{in.g};
            concrete_value_model<typename model_1::B_value> bv{in.b};

            ///Lin's type is dictated by gamma_to_linear
            auto lin_r = gamma_1::gamma::gamma_to_linear(rv, gamma_1());
            auto lin_g = gamma_1::gamma::gamma_to_linear(gv, gamma_1());
            auto lin_b = gamma_1::gamma::gamma_to_linear(bv, gamma_1());

            ///do not care about model
            if constexpr(std::is_same_v<space_1, space_2>)
            {
                out.r = gamma_2::gamma::linear_to_gamma(lin_r, gamma_2(), typename model_2::R_value()).v;
                out.g = gamma_2::gamma::linear_to_gamma(lin_g, gamma_2(), typename model_2::G_value()).v;
                out.b = gamma_2::gamma::linear_to_gamma(lin_b, gamma_2(), typename model_2::B_value()).v;
            }
            else
            {
                auto combo_convert = temporary::multiply(space_2::XYZ_to_linear, space_1::linear_to_XYZ);

                ///Todo: once real parameterised matrices are being used here (linear algebra proposal), and gamma_to_linear returns a value_type, parameterise the matrix
                ///based on that, for people who want to overload gamma_to_linear and linear_to_gamma
                auto vec = temporary::multiply(combo_convert, (temporary::vector_1x3){lin_r.v, lin_g.v, lin_b.v});

                decltype(lin_r) o_r{vec.a[0]};
                decltype(lin_g) o_g{vec.a[1]};
                decltype(lin_b) o_b{vec.a[2]};

                out.r = gamma_2::gamma::linear_to_gamma(o_r, gamma_2(), typename model_2::R_value()).v;
                out.g = gamma_2::gamma::linear_to_gamma(o_g, gamma_2(), typename model_2::G_value()).v;
                out.b = gamma_2::gamma::linear_to_gamma(o_b, gamma_2(), typename model_2::B_value()).v;
            }
        }

        alpha_convert(in, out);
    }

    ///generic RGB -> XYZ
    template<typename space, typename model, typename gamma, typename alpha_1, typename alpha_2>
    inline
    constexpr void color_convert(const basic_color<generic_RGB_space<space, gamma>, model, alpha_1>& in, basic_color<XYZ_space, XYZ_model, alpha_2>& out)
    {
        using type = space;

        concrete_value_model<typename model::R_value> rv{in.r};
        concrete_value_model<typename model::G_value> gv{in.g};
        concrete_value_model<typename model::B_value> bv{in.b};

        auto lin_r = gamma::gamma::gamma_to_linear(rv, gamma());
        auto lin_g = gamma::gamma::gamma_to_linear(gv, gamma());
        auto lin_b = gamma::gamma::gamma_to_linear(bv, gamma());

        auto vec = temporary::multiply(type::linear_to_XYZ, (temporary::vector_1x3){lin_r.v, lin_g.v, lin_b.v});

        model_convert_member<typename decltype(lin_r)::type, normalised_float_value_model>(vec.a[0], out.X);
        model_convert_member<typename decltype(lin_g)::type, normalised_float_value_model>(vec.a[1], out.Y);
        model_convert_member<typename decltype(lin_b)::type, normalised_float_value_model>(vec.a[2], out.Z);

        alpha_convert(in, out);
    }

    ///XYZ -> generic RGB
    template<typename space, typename model, typename gamma, typename alpha_1, typename alpha_2>
    inline
    constexpr void color_convert(const basic_color<XYZ_space, XYZ_model, alpha_1>& in, basic_color<generic_RGB_space<space, gamma>, model, alpha_2>& out)
    {
        using type = space;

        auto vec = temporary::multiply(type::XYZ_to_linear, (temporary::vector_1x3){in.X, in.Y, in.Z});

        concrete_value_model<normalised_float_value_model> c1{vec.a[0]};
        concrete_value_model<normalised_float_value_model> c2{vec.a[1]};
        concrete_value_model<normalised_float_value_model> c3{vec.a[2]};

        out.r = gamma::gamma::linear_to_gamma(c1, gamma(), typename model::R_value()).v;
        out.g = gamma::gamma::linear_to_gamma(c2, gamma(), typename model::G_value()).v;
        out.b = gamma::gamma::linear_to_gamma(c3, gamma(), typename model::B_value()).v;

        alpha_convert(in, out);
    }

    template<typename T, typename U, typename = void>
    struct has_direct_conversion_c : std::false_type{};

    template<typename T, typename U>
    struct has_direct_conversion_c<T, U, std::void_t<decltype(color_convert(std::declval<T>(), std::declval<U&>()))>> : std::true_type{};

    template<typename T, typename U>
    constexpr bool has_optimised_conversion()
    {
        return has_direct_conversion_c<T, U>::value;
    }

    template<typename space_1, typename model_1, typename alpha_1, typename space_2, typename model_2, typename alpha_2>
    constexpr bool is_same_parameterisation(const basic_color<space_1, model_1, alpha_1>& in, const basic_color<space_2, model_2, alpha_2>& out)
    {
        constexpr bool same_space = std::is_same_v<space_1, space_2>;
        constexpr bool same_model = std::is_same_v<model_1, model_2>;
        constexpr bool same_alpha = std::is_same_v<alpha_1, alpha_2>;

        return same_space && same_model && same_alpha;
    }

    template<typename T1, typename T2, typename... Args>
    inline
    constexpr void convert_impl(const T1& in, T2& out, Args&&... args)
    {
        if constexpr(is_same_parameterisation(in, out))
        {
            out = in;
            return;
        }

        if constexpr(has_optimised_conversion<decltype(in), decltype(out)>())
        {
            color_convert(in, out, std::forward<Args>(args)...);
        }
        else
        {
            basic_color<XYZ_space, XYZ_model, typename T1::alpha_type> intermediate;
            color_convert(in, intermediate, std::forward<Args>(args)...);
            color_convert(intermediate, out); ///TODO: Args2...
        }
    }

    template<typename destination, typename source, typename... T>
    inline
    constexpr destination convert(const source& in, T&&... args)
    {
        destination out;
        convert_impl(in, out, std::forward<T>(args)...);
        return out;
    }

    ///TODO:
    ///Runtime defined generic RGB space optimisation
    ///TODO:
    ///Is it legal to let a user overload this and stuff information into the struct?

    ///Details on runtime RGB stuff
    ///1. A naive conversion is very easy to do.
    ///Take a colour, apply the inverse of the gamma function with dynamic parameterisation
    ///multiply that with its toXYZ matrix
    ///multiply that with the destination fromXYZ matrix
    ///apply destination transform

    ///Optimisations that can be applied:
    ///1. Do nothing. Code will run much slower than static spaces
    ///2. When converting from floating point model colour space to floating point model colour space, the intermediate matrices can be combined, saving a multiplication
    ///3. When converting from ints to floating point, a lookup table could be used, which is as large as the start space - extremely fast
    ///4. When converting from floating point to ints, a lookup table can be used - very fast, though requires a larger lookup table than above and harder to get right

    ///If the set of optimisations is not the same as the set of optimisations applied by colour spaces with known compile time parameters (which are very easy to do)
    ///The following will result
    ///1. Implementation divergence
    ///2. Runtime vs compile time colour space result divergence, which is extremely confusing
    ///This is by far not the common case, however it is somewhat suboptimal

    ///For all the optimisation strategies, there is a tradeoff in space to store the data, and in some cases time to create the lookup tables
    ///This means that we may want some sort of converter state object, that holds 'misc' arbitrary state

    ///The problem with the above is that it complicates the general case, as for compile time colour spaces you have to do
    ///color::converter_state<src_space, dst_space> ccs(optimisation_type); dst_space res = ccs.convert(src_col);

    ///Even though ccs will be empty, because the optimisation will be dynamically calculated at compile time in ccs.convert, so no stack space is used assuming [[no_unique_address]]

    ///On the other hand, doing it like this does completely nail down implementation divergence, and provides a uniform API
    ///Would have to ensure that users didn't specify wrong combinations, eg float -> float table lookup

    ///These issues all become more pronounced with ICC support planned in the future, as converting to and from a device dependent colour space is likely fairly expensive
    ///but could be implement with decent precision and good performance with eg a 16 bit -> 16 bit table lookup in a lot of cases, once generated (expensive!)
    ///Fully custom user stuff could easily have int -> float or int -> int support added as a simple lookup. Float -> int support is significantly harder but theoretically doable
    ///Float -> float automatic matrix combination for non RGB derived types that still somehow have a valid matrix combination step may be a sore spot, but given the extreme rarity of this case, overloading the converter object and storing it yourself may be acceptable

    ///All in all, a bit of a mess, but a solvable one

    ///So: Tabulation is only necessary in the case of gamma -> linear. Tabulating 24bit rgb -> something else would be 16MB * sizeof(destination) ram... which isn't totally unacceptable in some cases

    /*template<typename destination, typename source, typename... T>
    struct connector
    {
        std::tuple<T...> custom_data;

        template<typename... U>
        connector(U&&... in) : custom_data(std::forward<U>(in)...){}

        constexpr destination convert(const source& in)
        {
            auto cvt = [&](auto&&... params)
            {
                return color::convert<destination, source, T...>(in, std::forward<decltype(params)>(params)...);
            };

            return std::apply(cvt, custom_data);
        }
    };

    template<typename destination, typename source, typename... T>
    connector<destination, source, T...> make_connector(T&&... args)
    {
        connector<destination, source, T...> ret(std::forward<T>(args)...);
        return ret;
    }*/

    ///TODO: reinterpret_color_space
    ///Should ignore colour space, but not ignore colour model

    ///TODO: complete overload set
    template<size_t N, typename T, typename U, typename V>
    constexpr std::tuple_element_t<N, color::RGB_model<T, U, V>>& get(color::RGB_model<T, U, V>& v) noexcept
    {
        if constexpr(N == 0)
            return v.r;

        if constexpr(N == 1)
            return v.g;

        if constexpr(N == 2)
            return v.b;
    }

    template<size_t N, typename T, typename U, typename V>
    constexpr std::tuple_element_t<N, color::RGB_model<T, U, V>>&& get(color::RGB_model<T, U, V>&& v) noexcept
    {
        if constexpr(N == 0)
            return std::move(v.r);

        if constexpr(N == 1)
            return std::move(v.g);

        if constexpr(N == 2)
            return std::move(v.b);
    }

    template<size_t N, typename T, typename U, typename V>
    constexpr const std::tuple_element_t<N, color::RGB_model<T, U, V>>&& get(const color::RGB_model<T, U, V>&& v) noexcept
    {
        if constexpr(N == 0)
            return std::move(v.r);

        if constexpr(N == 1)
            return std::move(v.g);

        if constexpr(N == 2)
            return std::move(v.b);
    }

    template<size_t N, typename T, typename U, typename V>
    constexpr const std::tuple_element_t<N, color::RGB_model<T, U, V>>& get(const color::RGB_model<T, U, V>& v) noexcept
    {
        if constexpr(N == 0)
            return v.r;

        if constexpr(N == 1)
            return v.g;

        if constexpr(N == 2)
            return v.b;
    }

    template<size_t N, typename T>
    constexpr std::tuple_element_t<N, color::alpha_model<T>>& get(color::alpha_model<T>& v) noexcept
    {
        if constexpr(N == 0)
            return v.a;
    }

    template<size_t N, typename T>
    constexpr std::tuple_element_t<N, color::alpha_model<T>>&& get(color::alpha_model<T>&& v) noexcept
    {
        if constexpr(N == 0)
            return std::move(v.a);
    }

    template<size_t N, typename T>
    constexpr const std::tuple_element_t<N, color::alpha_model<T>>& get(const color::alpha_model<T>& v) noexcept
    {
        if constexpr(N == 0)
            return v.a;
    }

    template<size_t N>
    constexpr void get(const color::alpha_model<void>& v) noexcept
    {

    }

    template<size_t N>
    constexpr void get(const color::alpha_model<void>&& v) noexcept
    {

    }

    /*template<size_t N, typename T, typename V>
    constexpr auto get(color::composite_model<T, V>& v)
    {
        if constexpr(N >= std::tuple_size_v<T>)
            return get<N - std::tuple_size_v<T>>((V&)v);

        return get<N>((T&)v);
    }

    template<size_t N, typename T, typename V>
    constexpr auto get(const color::composite_model<T, V>& v)
    {
        if constexpr(N >= std::tuple_size_v<T>)
            return get<N - std::tuple_size_v<T>>((V&)v);

        return get<N>((T&)v);
    }*/
}

template<typename T, typename U, typename V>
struct std::tuple_size<color::RGB_model<T, U, V>> : std::integral_constant<std::size_t, 3>{};

template<typename T>
struct std::tuple_size<color::alpha_model<T>> : std::integral_constant<std::size_t, 1>{};

template<>
struct std::tuple_size<color::alpha_model<void>> : std::integral_constant<std::size_t, 0>{};

template<typename T, typename U>
struct std::tuple_size<color::composite_model<T, U>> : std::integral_constant<std::size_t, std::tuple_size_v<T> + std::tuple_size_v<U>>{};

template<typename T, typename U, typename V>
struct std::tuple_element<0, color::RGB_model<T, U, V>> {using type = typename T::type;};

template<typename T, typename U, typename V>
struct std::tuple_element<1, color::RGB_model<T, U, V>> {using type = typename U::type;};

template<typename T, typename U, typename V>
struct std::tuple_element<2, color::RGB_model<T, U, V>> {using type = typename V::type;};

template<typename T>
struct std::tuple_element<0, color::alpha_model<T>> {using type = typename T::type;};

template<size_t I, typename T, typename U>
struct std::tuple_element<I, color::composite_model<T, U>>
{
    using type =
    std::conditional_t<
        I >= std::tuple_size_v<T>,
        std::tuple_element_t<I - std::tuple_size_v<T>, U>,
        std::tuple_element_t<I, T>>;
};

#endif // COLOR_HPP_INCLUDED
