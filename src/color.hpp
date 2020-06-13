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
#include <cstdint>

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

        constexpr
        vector_1x3(float v1, float v2, float v3)
        {
            a[0] = v1;
            a[1] = v2;
            a[2] = v3;
        }

        constexpr
        vector_1x3(){}
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
    constexpr void model_convert_member(const typename T1::type& in, typename T2::type& out)
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

    struct sRGB_gamma_parameters
    {
        static constexpr float transfer_alpha = 1.055;
        static constexpr float transfer_beta = 0.0031308;
        static constexpr float transfer_gamma = 12/5.f;
        static constexpr float transfer_delta = 12.92;
        static constexpr float transfer_bdelta = 0.04045;
    };

    template<typename T>
    struct type_base_t
    {
        T&& t;

        constexpr type_base_t(T&& in) : t(std::forward<T>(in)){}

        constexpr bool has_value(){return true;}

        T&& value(){return std::forward<T>(t);}
    };

    template<>
    struct type_base_t<void>
    {
        constexpr type_base_t(){}

        constexpr bool has_value(){return false;}
    };

    template<typename T>
    using nullarg = type_base_t<T>;

    template<typename T>
    struct arg_src_t : type_base_t<T>
    {

    };

    template<typename T>
    struct arg_dst_t : type_base_t<T>
    {

    };

    template<typename T>
    struct tf_src_t : type_base_t<T>
    {

    };

    template<typename T>
    struct tf_dst_t : type_base_t<T>
    {

    };

    namespace transfer_function
    {
        template<typename params>
        struct gamma_static
        {
            template<typename nonlinear, typename linear = normalised_float_value_model>
            static constexpr
            typename linear::type to_linear(typename nonlinear::type in_component)
            {
                float component = 0;
                model_convert_member<nonlinear, normalised_float_value_model>(in_component, component);

                if(component <= params::transfer_bdelta)
                    return component / params::transfer_delta;
                else
                    return std::pow((component + params::transfer_alpha - 1) / params::transfer_alpha, params::transfer_gamma);
            }

            template<typename nonlinear, typename linear = normalised_float_value_model>
            static constexpr
            typename nonlinear::type from_linear(typename linear::type in_component)
            {
                float my_val = 0;

                if(in_component <= params::transfer_beta)
                    my_val = in_component * params::transfer_delta;
                else
                    my_val = params::transfer_alpha * std::pow(in_component, 1/params::transfer_gamma) - (params::transfer_alpha - 1);

                typename nonlinear::type ret = typename nonlinear::type();
                model_convert_member<normalised_float_value_model, nonlinear>(my_val, ret);
                return ret;
            }
        };

        struct none
        {
            template<typename nonlinear, typename linear = normalised_float_value_model>
            static constexpr
            typename linear::type to_linear(typename nonlinear::type in_component)
            {
                typename linear::type component = typename linear::type();
                model_convert_member<nonlinear, linear>(in_component, component);
                return component;
            }

            template<typename nonlinear, typename linear = normalised_float_value_model>
            static constexpr
            typename nonlinear::type from_linear(typename linear::type in_component)
            {
                typename nonlinear::type component = typename nonlinear::type();
                model_convert_member<linear, nonlinear>(in_component, component);
                return component;
            }
        };

        struct sRGB_gamma
        {
            template<typename nonlinear, typename linear = normalised_float_value_model>
            static constexpr
            typename linear::type to_linear(typename nonlinear::type in_component)
            {
                return gamma_static<sRGB_gamma_parameters>::to_linear<nonlinear, linear>(in_component);
            }

            template<typename nonlinear, typename linear = normalised_float_value_model>
            static constexpr
            typename nonlinear::type from_linear(typename linear::type in_component)
            {
                return gamma_static<sRGB_gamma_parameters>::from_linear<nonlinear, linear>(in_component);
            }
        };
    }

    ///Specialise this to define a transfer functions linear type, TF = transfer function, nonlinear = input type
    template<typename TF, typename nonlinear>
    struct linear_type
    {
        using value = normalised_float_value_model;
    };

    template<typename TF, typename nonlinear>
    using linear_type_v = typename linear_type<TF, nonlinear>::value;

    struct sRGB_parameters
    {
        ///These are not mandatory for an RGB type
        static constexpr chromaticity R{0.64, 0.33};
        static constexpr chromaticity G{0.30, 0.60};
        static constexpr chromaticity B{0.15, 0.06};
        static constexpr chromaticity W = illuminant::CIE1931::D65;

        ///TODO: need to use rounded sRGB parameters from the spec rather than calculating them
        ///These *are* mandatory for an RGB type
        static constexpr temporary::matrix_3x3 linear_to_XYZ = get_linear_RGB_to_XYZ(R, G, B, W);
        static constexpr temporary::matrix_3x3 XYZ_to_linear = linear_to_XYZ.invert();
    };

    struct XYZ_parameters
    {
        static constexpr temporary::matrix_3x3 linear_to_XYZ = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        static constexpr temporary::matrix_3x3 XYZ_to_linear = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    };

    /*struct sRGB_transfer_parameters
    {
        static constexpr float transfer_alpha = 1.055;
        static constexpr float transfer_beta = 0.0031308;
        static constexpr float transfer_gamma = 12/5.f;
        static constexpr float transfer_delta = 12.92;
        static constexpr float transfer_bdelta = 0.04045;

        template<typename T, typename U>
        using transfer_function = typename transfer_function<T, U>::gamma;
    };

    struct empty_transfer_parameters
    {
        template<typename T, typename U>
        using transfer_function = typename transfer_function<T, U>::none;
    };*/

    template<typename T, typename U>
    struct generic_RGB_space : basic_color_space
    {
        using RGB_parameters = T;

        using transfer_function_parameters = U;
    };

    using sRGB_space = generic_RGB_space<sRGB_parameters, transfer_function::sRGB_gamma>;
    ///'linear color' is really linear sRGB specifically
    using linear_sRGB_space = generic_RGB_space<sRGB_parameters, transfer_function::none>;

    /*struct XYZ_space : basic_color_space
    {

    };*/

    using XYZ_space = generic_RGB_space<XYZ_parameters, transfer_function::none>;

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
        /*float X = 0;
        float Y = 0;
        float Z = 0;*/

        union
        {
            float r;
            float X;
        };

        union
        {
            float g;
            float Y;
        };

        union
        {
            float b;
            float Z;
        };

        constexpr XYZ_model() : r(0), g(0), b(0)
        {

        }

        using R_value = unnormalised_float_value_model;
        using G_value = unnormalised_float_value_model;
        using B_value = unnormalised_float_value_model;
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
        constexpr XYZ(float _X, float _Y, float _Z){r = _X; g = _Y; b = _Z;}
        constexpr XYZ(){}
    };

    template<typename T1, typename T2>
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

    ///direct conversion between two arbitrary RGB space
    template<typename space_1, typename model_1, typename gamma_1, typename alpha_1, typename space_2, typename model_2, typename gamma_2, typename alpha_2, typename tf_args_1 = void, typename tf_args_2 = void>
    constexpr void color_convert(const basic_color<generic_RGB_space<space_1, gamma_1>, model_1, alpha_1>& in, basic_color<generic_RGB_space<space_2, gamma_2>, model_2, alpha_2>& out,
                                 arg_src_t<void> a1 = arg_src_t<void>(), arg_dst_t<void> a2 = arg_dst_t<void>(),
                                 tf_src_t<tf_args_1> tf_data_1 = tf_src_t<tf_args_1>(), tf_dst_t<tf_args_2> tf_data_2 = tf_dst_t<tf_args_2>())
    {
        ///different models and alpha
        if constexpr(std::is_same_v<space_1, space_2> && std::is_same_v<gamma_1, gamma_2>)
        {
            model_convert(in, out);
        }
        else
        {
            using linear_R_t = linear_type_v<gamma_1, typename model_1::R_value>;
            using linear_G_t = linear_type_v<gamma_1, typename model_1::G_value>;
            using linear_B_t = linear_type_v<gamma_1, typename model_1::B_value>;

            using nonlinear_1_R_t = typename model_1::R_value;
            using nonlinear_1_G_t = typename model_1::G_value;
            using nonlinear_1_B_t = typename model_1::B_value;

            using nonlinear_2_R_t = typename model_2::R_value;
            using nonlinear_2_G_t = typename model_2::G_value;
            using nonlinear_2_B_t = typename model_2::B_value;

            typename linear_R_t::type lin_r = typename linear_R_t::type();
            typename linear_G_t::type lin_g = typename linear_G_t::type();
            typename linear_B_t::type lin_b = typename linear_B_t::type();

            if constexpr(!std::is_same_v<void, tf_args_1>)
            {
                lin_r = gamma_1::template to_linear<nonlinear_1_R_t, linear_R_t>(in.r, tf_data_1.value());
                lin_g = gamma_1::template to_linear<nonlinear_1_G_t, linear_G_t>(in.g, tf_data_1.value());
                lin_b = gamma_1::template to_linear<nonlinear_1_B_t, linear_B_t>(in.b, tf_data_1.value());
            }
            else
            {
                lin_r = gamma_1::template to_linear<nonlinear_1_R_t, linear_R_t>(in.r);
                lin_g = gamma_1::template to_linear<nonlinear_1_G_t, linear_G_t>(in.g);
                lin_b = gamma_1::template to_linear<nonlinear_1_B_t, linear_B_t>(in.b);
            }

            if constexpr(!std::is_same_v<space_1, space_2>)
            {
                auto combo_convert = temporary::multiply(space_2::XYZ_to_linear, space_1::linear_to_XYZ);

                ///Todo: once real parameterised matrices are being used here (linear algebra proposal), and gamma_to_linear returns a value_type, parameterise the matrix
                ///based on that, for people who want to overload gamma_to_linear and linear_to_gamma
                auto vec = temporary::multiply(combo_convert, temporary::vector_1x3{lin_r, lin_g, lin_b});

                lin_r = vec.a[0];
                lin_g = vec.a[1];
                lin_b = vec.a[2];
            }

            if constexpr(!std::is_same_v<void, tf_args_2>)
            {
                out.r = gamma_2::template from_linear<nonlinear_2_R_t, linear_R_t>(lin_r, tf_data_2.value());
                out.g = gamma_2::template from_linear<nonlinear_2_G_t, linear_G_t>(lin_g, tf_data_2.value());
                out.b = gamma_2::template from_linear<nonlinear_2_B_t, linear_B_t>(lin_b, tf_data_2.value());
            }
            else
            {
                out.r = gamma_2::template from_linear<nonlinear_2_R_t, linear_R_t>(lin_r);
                out.g = gamma_2::template from_linear<nonlinear_2_G_t, linear_G_t>(lin_g);
                out.b = gamma_2::template from_linear<nonlinear_2_B_t, linear_B_t>(lin_b);
            }
        }

        alpha_convert(in, out);
    }

    ///generic RGB -> XYZ
    template<typename space, typename model, typename trans, typename alpha_1, typename alpha_2, typename tf_args_1 = void>
    constexpr void color_convert(const basic_color<generic_RGB_space<space, trans>, model, alpha_1>& in, basic_color<XYZ_space, XYZ_model, alpha_2>& out, arg_src_t<void> a1 = arg_src_t<void>(), tf_src_t<tf_args_1> tf_data_1 = tf_src_t<tf_args_1>())
    {
        using type = space;

        using nonlinear_1_R_t = typename model::R_value;
        using nonlinear_1_G_t = typename model::G_value;
        using nonlinear_1_B_t = typename model::B_value;

        using linear_R_t = linear_type_v<trans, typename model::R_value>;
        using linear_G_t = linear_type_v<trans, typename model::G_value>;
        using linear_B_t = linear_type_v<trans, typename model::B_value>;

        auto lin_r = trans::template to_linear<nonlinear_1_R_t, linear_R_t>(in.r);
        auto lin_g = trans::template to_linear<nonlinear_1_G_t, linear_G_t>(in.g);
        auto lin_b = trans::template to_linear<nonlinear_1_B_t, linear_B_t>(in.b);

        auto vec = temporary::multiply(type::linear_to_XYZ, temporary::vector_1x3{lin_r, lin_g, lin_b});

        model_convert_member<linear_R_t, normalised_float_value_model>(vec.a[0], out.X);
        model_convert_member<linear_G_t, normalised_float_value_model>(vec.a[1], out.Y);
        model_convert_member<linear_B_t, normalised_float_value_model>(vec.a[2], out.Z);

        alpha_convert(in, out);
    }

    ///XYZ -> generic RGB
    template<typename space, typename model, typename trans, typename alpha_1, typename alpha_2, typename tf_args_1 = void>
    constexpr void color_convert(const basic_color<XYZ_space, XYZ_model, alpha_1>& in, basic_color<generic_RGB_space<space, trans>, model, alpha_2>& out, arg_dst_t<void> a1 = arg_dst_t<void>(), tf_dst_t<tf_args_1> tf_data_1 = tf_dst_t<tf_args_1>())
    {
        auto vec = temporary::multiply(space::XYZ_to_linear, temporary::vector_1x3{in.X, in.Y, in.Z});

        /*out.r = trans::template transfer_function<typename model::R_transfer::nonlinear_t, typename model::R_transfer::linear_t>::from_linear(vec.a[0], trans());
        out.g = trans::template transfer_function<typename model::G_transfer::nonlinear_t, typename model::G_transfer::linear_t>::from_linear(vec.a[1], trans());
        out.b = trans::template transfer_function<typename model::B_transfer::nonlinear_t, typename model::B_transfer::linear_t>::from_linear(vec.a[2], trans());*/

        using nonlinear_1_R_t = typename model::R_value;
        using nonlinear_1_G_t = typename model::G_value;
        using nonlinear_1_B_t = typename model::B_value;

        out.r = trans::template from_linear<nonlinear_1_R_t, normalised_float_value_model>(vec.a[0]);
        out.g = trans::template from_linear<nonlinear_1_G_t, normalised_float_value_model>(vec.a[1]);
        out.b = trans::template from_linear<nonlinear_1_B_t, normalised_float_value_model>(vec.a[2]);

        alpha_convert(in, out);
    }

    template<typename T, typename U, typename = void>
    struct has_direct_conversion_c : std::false_type{};

    template<typename T, typename U>
    struct has_direct_conversion_c<T, U, std::void_t<decltype(color_convert(std::declval<T>(), std::declval<U&>()))>> : std::true_type{};

    template<typename T, typename U, typename A1, typename A2, typename A3, typename A4, typename = void>
    struct has_6arg_conversion_c : std::false_type{};

    template<typename T, typename U, typename A1, typename A2, typename A3, typename A4>
    struct has_6arg_conversion_c<T, U, A1, A2, A3, A4, std::void_t<decltype(color_convert(std::declval<T>(), std::declval<U&>(),
                                                                                          std::declval<A1>(), std::declval<A2>(), std::declval<A3>(), std::declval<A4>()))>> : std::true_type{};

    template<typename T, typename U, typename A1, typename A2, typename = void>
    struct has_2arg_conversion_c : std::false_type{};

    template<typename T, typename U, typename A1, typename A2>
    struct has_2arg_conversion_c<T, U, A1, A2, std::void_t<decltype(color_convert(std::declval<T>(), std::declval<U&>(),
                                                                                  std::declval<A1>(), std::declval<A2>()))>> : std::true_type{};

    template<typename T, typename U>
    constexpr bool has_optimised_conversion()
    {
        return has_direct_conversion_c<T, U>::value;
    }

    template<typename... T>
    constexpr bool has_6arg_conversion()
    {
        return has_6arg_conversion_c<T...>::value;
    }

    template<typename... T>
    constexpr bool has_2arg_conversion()
    {
        return has_2arg_conversion_c<T...>::value;
    }

    template<typename space_1, typename model_1, typename alpha_1, typename space_2, typename model_2, typename alpha_2>
    constexpr bool is_same_parameterisation(const basic_color<space_1, model_1, alpha_1>& in, const basic_color<space_2, model_2, alpha_2>& out)
    {
        (void)in;
        (void)out;

        constexpr bool same_space = std::is_same_v<space_1, space_2>;
        constexpr bool same_model = std::is_same_v<model_1, model_2>;
        constexpr bool same_alpha = std::is_same_v<alpha_1, alpha_2>;

        return same_space && same_model && same_alpha;
    }

    template<typename T, template <typename...> typename arg_template>
    struct is_specialisation : std::false_type {};

    template<template<typename...> typename arg_template, typename... Args>
    struct is_specialisation<arg_template<Args...>, arg_template> : std::true_type {};

    template<template<typename...> typename arg_template, typename T>
    constexpr auto tuple_get_arg(T&& arg)
    {
        if constexpr(is_specialisation<T, arg_template>::value)
        {
            return std::forward_as_tuple(std::forward<T>(arg));
        }
        else
        {
            return std::tuple<>();
        }
    }

    template<template<typename...> typename arg_template, typename Or, typename... T>
    constexpr auto tuple_type_or(std::tuple<T...>&& arg)
    {
        if constexpr(sizeof...(T) == 0)
        {
            return std::tuple<arg_template<Or>>(arg_template<Or>());
        }
        else
        {
            return arg;
        }
    }

    template<template<typename...> typename arg_template, typename... T>
    constexpr auto tuple_arg_construct(T&&... args)
    {
        return tuple_type_or<arg_template, void>(std::tuple_cat(tuple_get_arg<arg_template>(args)...));
    }

    template<typename... T>
    constexpr bool tuple_6arg_exists(const std::tuple<T...>&)
    {
        return has_6arg_conversion<T...>();
    }

    template<typename... T>
    constexpr bool tuple_2arg_exists(const std::tuple<T...>&)
    {
        return has_2arg_conversion<T...>();
    }

    template<typename T1, typename T2, typename... Args>
    constexpr void convert_impl(const T1& in, T2& out, Args&&... args)
    {
        if constexpr(std::is_same_v<T1, T2>)
        {
            out = in;
            return;
        }

        if constexpr(has_optimised_conversion<decltype(in), decltype(out)>())
        {
            std::tuple<const T1&, T2&> tup(in, out);

            auto arg_tup = std::tuple_cat(tup,
                                          tuple_arg_construct<arg_src_t>(args...),
                                          tuple_arg_construct<arg_dst_t>(args...),
                                          tuple_arg_construct<tf_src_t>(args...),
                                          tuple_arg_construct<tf_dst_t>(args...));

            static_assert(std::tuple_size_v<decltype(arg_tup)> == 6);

            if constexpr(tuple_6arg_exists(arg_tup))
            {
                std::apply([&](auto&&... args)
                {
                    color_convert(args...);
                }, arg_tup);
            }
            else
            {
                std::apply([&](auto&&... args)
                {
                    color_convert(args...);
                }, tup);
            }
        }
        else
        {
            basic_color<XYZ_space, XYZ_model, typename T1::alpha_type> val;

            std::tuple<const T1&, basic_color<XYZ_space, XYZ_model, typename T1::alpha_type>&> tup(in, val);
            std::tuple<const basic_color<XYZ_space, XYZ_model, typename T1::alpha_type>&, T2&> tup2(val, out);

            auto arg_tup_1 = std::tuple_cat(tup,
                                            tuple_arg_construct<arg_src_t>(args...),
                                            tuple_arg_construct<tf_src_t>(args...));

            if constexpr(tuple_2arg_exists(arg_tup_1))
            {
                std::apply([&](auto&&... args)
                {
                    color_convert(args...);
                }, arg_tup_1);
            }
            else
            {
                std::apply([&](auto&&... args)
                {
                    color_convert(args...);
                }, tup);
            }

            auto arg_tup_2 = std::tuple_cat(tup2,
                                            tuple_arg_construct<arg_dst_t>(args...),
                                            tuple_arg_construct<tf_dst_t>(args...));

            if constexpr(tuple_2arg_exists(arg_tup_2))
            {
                std::apply([&](auto&&... args)
                {
                    color_convert(args...);
                }, arg_tup_2);
            }
            else
            {
                std::apply([&](auto&&... args)
                {
                    color_convert(args...);
                }, tup2);
            }
        }
    }

    template<typename destination, typename source, typename... T>
    constexpr destination convert(const source& in, T&&... args)
    {
        destination out = destination();
        convert_impl(in, out, std::forward<T>(args)...);
        return out;
    }
}

#endif // COLOR_HPP_INCLUDED
