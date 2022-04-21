#pragma once

template<typename T>
struct Vec2D {
    T x, y;
};

template<typename T>
struct Vec3D {
    union {
        T data[3];
        struct { T x, y, z; };
        struct { T r, g, b; };
    };
};