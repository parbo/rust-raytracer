extern crate vecmath as vecmath_lib;

pub type Vec3 = self::vecmath_lib::Vector3<f64>;

pub use self::vecmath_lib::vec3_add as add;
pub use self::vecmath_lib::vec3_cross as cross;
pub use self::vecmath_lib::vec3_dot as dot;
pub use self::vecmath_lib::vec3_neg as neg;
pub use self::vecmath_lib::vec3_normalized as normalize;
pub use self::vecmath_lib::vec3_scale as mul;
pub use self::vecmath_lib::vec3_square_len as square_length;
pub use self::vecmath_lib::vec3_sub as sub;

pub fn length(v: Vec3) -> f64 {
    square_length(v).sqrt()
}
