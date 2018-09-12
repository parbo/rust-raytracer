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

// class SpotLight(object):
//     def __init__(self, pos, at, c, cutoff, exp):
//         self.pos = pos
//         self.at = at
//         self.d = normalize(sub(self.at, self.pos))
//         self.color = c
//         self.cutoff = cutoff
//         self.coscutoff = math.cos(math.radians(self.cutoff))
//         self.exp = exp

//     def get_direction(self, pos):
//         d = sub(self.pos, pos)
//         dl = length(d)
//         return mul(d, 1.0 / dl), length(d)

//     def get_intensity(self, pos):
//         q = sub(self.pos, pos)
//         dsq = dot(q, q)
//         invlend = 1 / math.sqrt(dsq)
//         cosangle = -dot(q, self.d) * invlend
//         if cosangle < self.coscutoff:
//             return (0.0, 0.0, 0.0)
//         i = pow(dot(self.d, mul(d, -invlend)), self.exp)
//         return mul(self.color, i * 100.0 / (99.0 + dsq))
