#pragma once

#include <iostream>
#include "xtensor/xoffset_view.hpp"

namespace xt
{
    template <class T>
    struct quaternion
    {
        using value_type = T;
        T w, x, y, z;

        quaternion() = default;
        quaternion(const quaternion<T>&) = default;
        quaternion(const T& _w, const T& _x, const T& _y, const T& _z)
            : w(_w), x(_x), y(_y), z(_z) {};
        quaternion(const T& in) {
            w = in;
            x = y = z = 0;
        }
        bool operator==(const quaternion<T>& rhs) const {
            return (w == rhs.w) && (x == rhs.x) && (y == rhs.y) && (z == rhs.z);
        };

        // the Hamilton product
        quaternion operator*(const quaternion& rhs) const {
            return {
                w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z,
                w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
                w * rhs.y + y * rhs.w - x * rhs.z + z * rhs.x,
                w * rhs.z + z * rhs.w + x * rhs.y - y * rhs.x
            };
        }

        quaternion operator+(const quaternion& rhs) const {
            return {w + rhs.w, x + rhs.x, y + rhs.y, z + rhs.z};
        }

        template <class A, std::enable_if_t<std::is_arithmetic_v<A>, int> = 0>
        quaternion operator+(const A& a) const {
            return {w + a, x, y, z};
        }

        template <class A, std::enable_if_t<std::is_arithmetic_v<A>, int> = 0>
        quaternion operator-(const A& a) const {
            return {w - a, x, y, z};
        }

        quaternion operator-(const quaternion& rhs) const {
            return {w - rhs.w, x - rhs.x, y - rhs.y, z - rhs.z};
        }

        template <class A, std::enable_if_t<std::is_arithmetic_v<A>, int> = 0>
        quaternion operator*(const A& a) const {
            return {w*a, x*a, y*a, z*a};
        }

        template <class A, std::enable_if_t<std::is_arithmetic_v<A>, int> = 0>
        quaternion operator/(const A& a) const {
            return {w/a, x/a, y/a, z/a};
        }

        void operator=(const T& rhs) {
            w = rhs;
            x = y = z = 0;
        }

        quaternion conj() const {
            return {w, -x, -y, -z};
        }
    };

    template <class A, class T, std::enable_if_t<std::is_arithmetic_v<A>, int> = 0>
    quaternion<T> operator+(const A& a, const quaternion<T>& q) {
        return {q.w + a, q.x, q.y, q.z};
    }

    template <class A, class T, std::enable_if_t<std::is_arithmetic_v<A>, int> = 0>
    quaternion<T> operator-(const A& a, const quaternion<T>& q) {
        return {a - q.w, -q.x, -q.y, -q.z};
    }

    template <class T>
    std::ostream& operator<<(std::ostream& o, const quaternion<T>& q) {
        o << "Q: " << q.w << ", " << q.x << ", " << q.y << ", " << q.z << "\n";
        return o;
    }

    template <std::size_t O, class X>
    auto offset_view_impl(X&& x) {
        using value_type = typename std::decay_t<X>::value_type::value_type;
        return xoffset_view<xclosure_t<X>, value_type, O>(x);
    }

    #define offset_view(X, N) offset_view_impl<offsetof(typename decltype(X)::value_type, N)>(X)

    namespace math
    {
        namespace detail
        {
            template <class T>
            quaternion<T> conj_impl(const quaternion<T>& q){
                return q.conj();
            }
        }
    }
}

#include "xtensor/xcomplex.hpp"
