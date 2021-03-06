use vecmath::{identity, mmmul, mvmuly, transform, transpose, Mat4, Vec3};

#[derive(Clone)]
pub struct Transform {
    m: Mat4,
    inv_m: Mat4,
}

fn transform_point(m: Mat4, p: Vec3) -> Vec3 {
    let v4 = [p[0], p[1], p[2], 1.0];
    let v4res = transform(m, v4);
    [v4res[0], v4res[1], v4res[2]]
}

fn transform_vector(m: Mat4, p: Vec3) -> Vec3 {
    let v4 = [p[0], p[1], p[2], 0.0];
    let v4res = transform(m, v4);
    [v4res[0], v4res[1], v4res[2]]
}

impl Transform {
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        transform_point(self.m, p)
    }

    pub fn inv_transform_point(&self, p: Vec3) -> Vec3 {
        transform_point(self.inv_m, p)
    }

    pub fn inv_transform_vector(&self, p: Vec3) -> Vec3 {
        transform_vector(self.inv_m, p)
    }

    pub fn transform_normal(&self, v: Vec3) -> Vec3 {
        transform_vector(transpose(self.inv_m), v)
    }

    pub fn inv_transform_py(&self, v: Vec3) -> f64 {
        let [x, y, z] = v;
        mvmuly(self.inv_m, [x, y, z, 1.0])
    }

    pub fn scale(&mut self, sx: f64, sy: f64, sz: f64) {
        let sc = [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inv_sc = [
            [1.0 / sx, 0.0, 0.0, 0.0],
            [0.0, 1.0 / sy, 0.0, 0.0],
            [0.0, 0.0, 1.0 / sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        self.m = mmmul(sc, self.m);
        self.inv_m = mmmul(self.inv_m, inv_sc);
    }

    pub fn uscale(&mut self, s: f64) {
        self.scale(s, s, s);
    }

    pub fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
        let tr = [
            [1.0, 0.0, 0.0, tx],
            [0.0, 1.0, 0.0, ty],
            [0.0, 0.0, 1.0, tz],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inv_tr = [
            [1.0, 0.0, 0.0, -tx],
            [0.0, 1.0, 0.0, -ty],
            [0.0, 0.0, 1.0, -tz],
            [0.0, 0.0, 0.0, 1.0],
        ];
        self.m = mmmul(tr, self.m);
        self.inv_m = mmmul(self.inv_m, inv_tr);
    }

    pub fn rotatex(&mut self, d: f64) {
        let cosd = d.to_radians().cos();
        let sind = d.to_radians().sin();
        let rx = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosd, -sind, 0.0],
            [0.0, sind, cosd, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inv_rx = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosd, sind, 0.0],
            [0.0, -sind, cosd, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        self.m = mmmul(rx, self.m);
        self.inv_m = mmmul(self.inv_m, inv_rx);
    }

    pub fn rotatey(&mut self, d: f64) {
        let cosd = d.to_radians().cos();
        let sind = d.to_radians().sin();
        let ry = [
            [cosd, 0.0, sind, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sind, 0.0, cosd, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inv_ry = [
            [cosd, 0.0, -sind, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [sind, 0.0, cosd, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        self.m = mmmul(ry, self.m);
        self.inv_m = mmmul(self.inv_m, inv_ry);
    }

    pub fn rotatez(&mut self, d: f64) {
        let cosd = d.to_radians().cos();
        let sind = d.to_radians().sin();
        let rz = [
            [cosd, -sind, 0.0, 0.0],
            [sind, cosd, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let inv_rz = [
            [cosd, sind, 0.0, 0.0],
            [-sind, cosd, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        self.m = mmmul(rz, self.m);
        self.inv_m = mmmul(self.inv_m, inv_rz);
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
