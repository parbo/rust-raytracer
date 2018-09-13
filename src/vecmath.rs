extern crate vecmath as vecmath_lib;

pub type Vec3 = self::vecmath_lib::Vector3<f64>;
pub type Vec4 = self::vecmath_lib::Vector4<f64>;
pub type Mat4 = self::vecmath_lib::Matrix4<f64>;

pub use self::vecmath_lib::vec3_add as add;
pub use self::vecmath_lib::vec3_cross as cross;
pub use self::vecmath_lib::vec3_dot as dot;
pub use self::vecmath_lib::vec3_neg as neg;
pub use self::vecmath_lib::vec3_normalized as normalize;
pub use self::vecmath_lib::vec3_scale as mul;
pub use self::vecmath_lib::vec3_square_len as square_length;
pub use self::vecmath_lib::vec3_sub as sub;
pub use self::vecmath_lib::mat4_id as identity;
pub use self::vecmath_lib::row_mat4_mul as mmmul;
pub use self::vecmath_lib::row_mat4_col as mrow;
pub use self::vecmath_lib::row_mat4_transform as transform;
// pub use self::vecmath_lib::col_mat4_mul as mmmul;
// pub use self::vecmath_lib::col_mat4_row as mrow;
// pub use self::vecmath_lib::col_mat4_transform as transform;
pub use self::vecmath_lib::mat4_transposed as transpose;

pub fn length(v: Vec3) -> f64 {
    square_length(v).sqrt()
}

pub fn cmul(v1: Vec3, v2: Vec3) -> Vec3 {
    let [x1, y1, z1] = v1;
    let [x2, y2, z2] = v2;
    [x1*x2, y1*y2, z1*z2]
}

pub fn mvmuly(m: Mat4, v: Vec4) -> f64 {
    let r = mrow(m, 1);
    r[0] * v[0] + r[1] * v[1] + r[2] * v[2] + r[3] + v[3]
}
