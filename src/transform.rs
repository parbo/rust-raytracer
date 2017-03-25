use vecmath::{Mat4, identity};

#[derive(Clone)]
pub struct Transform {
    m: Mat4,
    inv_m: Mat4
}

impl Default for Transform {
    fn default() -> Transform {
        Transform { m: identity(), inv_m: identity() }
    }
}
