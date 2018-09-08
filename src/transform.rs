use vecmath::{Mat4, identity, Vec3, transform};

#[derive(Clone)]
pub struct Transform {
    m: Mat4,
    inv_m: Mat4,
}

impl Transform {
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let v4 = [p[0], p[1], p[2], 1.0];
        let v4res = transform(self.m, v4);
        [v4res[0], v4res[1], v4res[2]]
    }
}

impl Default for Transform {
    fn default() -> Transform {
        Transform {
            m: identity(),
            inv_m: identity(),
        }
    }
}
