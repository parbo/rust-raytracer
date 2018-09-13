use vecmath::{Vec3, normalize, neg, sub, length, mul, dot};

pub trait Light: LightClone {
    fn name(&self) -> &str;
    fn get_direction(&self, pos: Vec3) -> (Vec3, Option<f64>);
    fn get_intensity(&self, pos: Vec3) -> Vec3;
}

pub trait LightClone {
    fn clone_box(&self) -> Box<Light>;
}

impl<T> LightClone for T
    where T: 'static + Light + Clone
{
    fn clone_box(&self) -> Box<Light> {
        Box::new(self.clone())
    }
}

impl Clone for Box<Light> {
    fn clone(&self) -> Box<Light> {
        self.clone_box()
    }
}

#[derive(Clone)]
pub struct DirectionalLight {
    direction: Vec3,
    color: Vec3,
}

impl DirectionalLight {
    pub fn new(d: Vec3, c: Vec3) -> DirectionalLight {
        DirectionalLight {
            direction: normalize(neg(d)),
            color: c,
        }
    }
}

impl Light for DirectionalLight {
    fn name(&self) -> &str {
        "light"
    }
    fn get_direction(&self, _pos: Vec3) -> (Vec3, Option<f64>) {
        (self.direction, None)
    }
    fn get_intensity(&self, _pos: Vec3) -> Vec3 {
        self.color
    }
}

#[derive(Clone)]
pub struct PointLight {
    pos: Vec3,
    color: Vec3,
}

impl PointLight {
    pub fn new(p: Vec3, c: Vec3) -> PointLight {
        PointLight {
            pos: p,
            color: c,
        }
    }
}

impl Light for PointLight {
    fn name(&self) -> &str {
        "pointlight"
    }
    fn get_direction(&self, pos: Vec3) -> (Vec3, Option<f64>) {
        let d = sub(self.pos, pos);
        let dl = length(d);
        (mul(d, 1.0 / dl), Some(length(d)))
    }
    fn get_intensity(&self, pos: Vec3) -> Vec3 {
        let d = sub(self.pos, pos);
        let dsq = dot(d, d);
        mul(self.color, 100.0 / (99.0 + dsq))
    }
}

#[derive(Clone)]
pub struct SpotLight {
    pos: Vec3,
    at: Vec3,
    d: Vec3,
    color: Vec3,
    cutoff: f64,
    coscutoff: f64,
    exp: f64
}

impl SpotLight {
    pub fn new(pos: Vec3,
               at: Vec3,
               color: Vec3,
               cutoff: f64,
               exp: f64) -> SpotLight {
        let d = normalize(sub(at, pos));
        let coscutoff = cutoff.to_radians().cos();
        SpotLight {
            pos: pos, at: at, d: d, color: color, cutoff: cutoff, coscutoff: coscutoff, exp: exp
        }
    }
}

impl Light for SpotLight {
    fn name(&self) -> &str {
        "spotlight"
    }

    fn get_direction(&self, pos: Vec3) -> (Vec3, Option<f64>) {
        let d = sub(self.pos, pos);
        let dl = length(d);
        (mul(d, 1.0 / dl), Some(dl))
    }

    fn get_intensity(&self, pos: Vec3) -> Vec3 {
        let d = sub(self.pos, pos);
        let dsq = dot(d, d);
        let invlend = 1.0 / dsq.sqrt();
        let cosangle = -dot(d, self.d) * invlend;
        if cosangle < self.coscutoff {
            return [0.0, 0.0, 0.0];
        }
        let i = dot(self.d, mul(d, -invlend)).powf(self.exp);
        mul(self.color, i * 100.0 / (99.0 + dsq))
    }
}
