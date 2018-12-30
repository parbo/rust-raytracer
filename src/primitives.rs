use std::cmp::Ordering;
use std::error::Error as StdError;
use std::fmt;
use std::iter;
use std::mem;
use std::rc::Rc;
use std::sync::atomic::{self, AtomicUsize};
use transform::Transform;
use vecmath::{add, dot, length, mul, neg, normalize, sub, Vec3};

static PRIMITIVE_COUNTER: AtomicUsize = atomic::ATOMIC_USIZE_INIT;

#[derive(Debug, PartialEq, Copy)]
pub struct PrimitiveId(usize);

#[derive(Debug)]
pub enum PrimitivesError {
    InvalidFace(i64),
}

impl fmt::Display for PrimitivesError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.description())
    }
}

impl StdError for PrimitivesError {
    fn description(&self) -> &str {
        match *self {
            PrimitivesError::InvalidFace(_) => "InvalidFace",
        }
    }
}

impl PrimitiveId {
    fn new() -> Self {
        PrimitiveId(PRIMITIVE_COUNTER.fetch_add(1, atomic::Ordering::SeqCst))
    }
}

impl Clone for PrimitiveId {
    fn clone(&self) -> PrimitiveId {
        PrimitiveId::new()
    }
}

pub trait Primitive {
    fn get_surface(&self, opos: Vec3, face: i64) -> Result<(Vec3, f64, f64, f64), PrimitivesError>;
    fn get_normal(&self, p: Vec3, face: i64) -> Result<Vec3, PrimitivesError>;
    fn get_transform(&self) -> &Transform;
    fn get_mut_transform(&mut self) -> &mut Transform;
    fn transform_point(&self, p: Vec3) -> Vec3 {
        self.get_transform().transform_point(p)
    }
}

pub trait IntersectRay {
    fn find_primitive(&self, id: PrimitiveId) -> Option<&Primitive>;
    fn intersect(&self, raypos: Vec3, raydir: Vec3) -> Vec<Intersection>;
}

pub trait Node: NodeClone + IntersectRay {
    fn translate(&mut self, tx: f64, ty: f64, tz: f64);
    fn scale(&mut self, sx: f64, sy: f64, sz: f64);
    fn uscale(&mut self, s: f64);
    fn rotatex(&mut self, d: f64);
    fn rotatey(&mut self, d: f64);
    fn rotatez(&mut self, d: f64);
}

pub trait NodeClone {
    fn clone_box(&self) -> Box<Node>;
}

impl<T> NodeClone for T
where
    T: 'static + Node + Clone,
{
    fn clone_box(&self) -> Box<Node> {
        Box::new(self.clone())
    }
}

impl Clone for Box<Node> {
    fn clone(&self) -> Box<Node> {
        self.clone_box()
    }
}

impl<T: Primitive + IntersectRay + Clone + 'static> Node for T {
    fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
        self.get_mut_transform().translate(tx, ty, tz);
    }
    fn scale(&mut self, sx: f64, sy: f64, sz: f64) {
        self.get_mut_transform().scale(sx, sy, sz);
    }
    fn uscale(&mut self, s: f64) {
        self.get_mut_transform().uscale(s);
    }
    fn rotatex(&mut self, d: f64) {
        self.get_mut_transform().rotatex(d);
    }
    fn rotatey(&mut self, d: f64) {
        self.get_mut_transform().rotatey(d);
    }
    fn rotatez(&mut self, d: f64) {
        self.get_mut_transform().rotatez(d);
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum IntersectionType {
    Entry,
    Exit,
}

#[derive(Copy, Clone)]
pub struct Intersection {
    odistance: f64,
    pub distance: f64,
    rp: Vec3,
    rd: Vec3,
    pub primitive_id: PrimitiveId,
    pub t: IntersectionType,
    pub original_t: IntersectionType,
    pub face: i64, // Todo: maybe use a type instea
}

impl PartialOrd for Intersection {
    fn partial_cmp(&self, other: &Intersection) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl PartialEq for Intersection {
    fn eq(&self, other: &Intersection) -> bool {
        self.distance == other.distance
    }
}

impl Intersection {
    pub fn new(
        scale: f64,
        odistance: f64,
        rp: Vec3,
        rd: Vec3,
        primitive_id: PrimitiveId,
        t: IntersectionType,
        face: i64,
    ) -> Intersection {
        Intersection {
            odistance,
            distance: scale * odistance,
            rp,
            rd,
            primitive_id,
            t,
            original_t: t,
            face,
        }
    }

    pub fn switched(&self) -> bool {
        self.t != self.original_t
    }
    pub fn switch(&mut self, t: IntersectionType) {
        if self.t != t {
            self.t = t;
        }
    }

    pub fn get_opos(&self) -> Vec3 {
        add(self.rp, mul(self.rd, self.odistance))
    }
}

#[derive(Clone)]
pub struct Operator {
    obj1: Box<Node>,
    obj2: Box<Node>,
    rule: &'static Fn(bool, bool) -> bool,
}

fn union(a: bool, b: bool) -> bool {
    a || b
}

fn intersect(a: bool, b: bool) -> bool {
    a && b
}

fn difference(a: bool, b: bool) -> bool {
    a && !b
}

impl Operator {
    pub fn make_union(obj1: Box<Node>, obj2: Box<Node>) -> Operator {
        Operator {
            obj1,
            obj2,
            rule: &union,
        }
    }
    pub fn make_intersect(obj1: Box<Node>, obj2: Box<Node>) -> Operator {
        Operator {
            obj1,
            obj2,
            rule: &intersect,
        }
    }
    pub fn make_difference(obj1: Box<Node>, obj2: Box<Node>) -> Operator {
        Operator {
            obj1,
            obj2,
            rule: &difference,
        }
    }
}

impl Node for Operator {
    fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
        self.obj1.translate(tx, ty, tz);
        self.obj2.translate(tx, ty, tz);
    }
    fn scale(&mut self, sx: f64, sy: f64, sz: f64) {
        self.obj1.scale(sx, sy, sz);
        self.obj2.scale(sx, sy, sz);
    }
    fn uscale(&mut self, s: f64) {
        self.obj1.uscale(s);
        self.obj2.uscale(s);
    }
    fn rotatex(&mut self, d: f64) {
        self.obj1.rotatex(d);
        self.obj2.rotatex(d);
    }
    fn rotatey(&mut self, d: f64) {
        self.obj1.rotatey(d);
        self.obj2.rotatey(d);
    }
    fn rotatez(&mut self, d: f64) {
        self.obj1.rotatez(d);
        self.obj2.rotatez(d);
    }
}

impl IntersectRay for Operator {
    fn find_primitive(&self, id: PrimitiveId) -> Option<&Primitive> {
        if let Some(n) = self.obj1.find_primitive(id) {
            Some(n)
        } else if let Some(n) = self.obj2.find_primitive(id) {
            Some(n)
        } else {
            None
        }
    }
    // fn inside(&self, pos: Vec3) -> bool {
    //     (self.rule)(self.obj1.inside(pos), self.obj2.inside(pos))
    // }
    fn intersect(&self, raypos: Vec3, raydir: Vec3) -> Vec<Intersection> {
        let mut inside1 = 0;
        let mut inside2 = 0;
        // if self.obj1.inside(raypos) {
        //     inside1 = 1;
        // }
        // if self.obj2.inside(raypos) {
        //     inside2 = 1;
        // }
        let mut inside = (self.rule)(inside1 > 0, inside2 > 0);
        let obj1i = self.obj1.intersect(raypos, raydir);
        let obj2i = self.obj2.intersect(raypos, raydir);

        let mut intersections: Vec<(&Intersection, i32)> = obj1i
            .iter()
            .zip(iter::repeat(1))
            .chain(obj2i.iter().zip(iter::repeat(2)))
            .collect();
        intersections.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Less));

        let mut res = Vec::<Intersection>::new();
        let mut prevt = 0.0;
        for (ref i, obj) in intersections.iter() {
            match i.t {
                IntersectionType::Entry => {
                    if *obj == 1 {
                        inside1 += 1;
                    } else {
                        inside2 += 1;
                    }
                }
                IntersectionType::Exit => {
                    if *obj == 1 {
                        inside1 -= 1;
                    } else {
                        inside2 -= 1;
                    }
                }
            }

            let newinside = (self.rule)(inside1 > 0, inside2 > 0);
            if inside && !newinside {
                if (i.distance - prevt) < 1e-10 {
                    // remove infinitesimal intersections
                    // to avoid problem with difference of touching surfaces
                    res.pop();
                } else {
                    let mut ic = **i;
                    ic.switch(IntersectionType::Exit);
                    res.push(ic);
                }
            }
            if !inside && newinside {
                let mut ic = **i;
                ic.switch(IntersectionType::Entry);
                prevt = ic.distance;
                res.push(ic);
            }
            inside = newinside;
        }

        res
    }
}

type SurfaceFunction = Fn(i64, f64, f64) -> (Vec3, f64, f64, f64);

#[derive(Clone)]
pub struct PrimitiveCommon {
    transform: Transform,
    surface: Rc<Box<SurfaceFunction>>,
    id: PrimitiveId,
}

#[derive(Clone)]
pub struct Sphere(PrimitiveCommon);

impl Sphere {
    pub fn new(surface: Rc<Box<SurfaceFunction>>) -> Sphere {
        Sphere(PrimitiveCommon {
            transform: Default::default(),
            surface,
            id: PrimitiveId::new(),
        })
    }
}

impl IntersectRay for Sphere {
    fn find_primitive(&self, id: PrimitiveId) -> Option<&Primitive> {
        if id == self.0.id {
            Some(self)
        } else {
            None
        }
    }

    fn intersect(&self, raypos: Vec3, raydir: Vec3) -> Vec<Intersection> {
        let tr = &self.0.transform;
        let transformed_raydir = tr.inv_transform_vector(raydir);
        let scale = 1.0 / length(transformed_raydir);
        let normalized_transformed_raydir = mul(transformed_raydir, scale); // normalize
        let transformed_raypos = tr.inv_transform_point(raypos);
        let s = dot(neg(transformed_raypos), normalized_transformed_raydir);
        let lsq = dot(transformed_raypos, transformed_raypos);
        if s < 0.0 && lsq > 1.0 {
            return vec![];
        }
        let msq = lsq - s * s;
        if msq > 1.0 {
            return vec![];
        }
        let q = (1.0 - msq).sqrt();
        let mut t1 = s + q;
        let mut t2 = s - q;
        if t1 > t2 {
            mem::swap(&mut t1, &mut t2);
        }
        let mut ts = vec![];
        if t1 > 0.0 {
            ts.push(Intersection::new(
                scale,
                t1,
                transformed_raypos,
                normalized_transformed_raydir,
                self.0.id,
                IntersectionType::Entry,
                0,
            ));
        }
        if t2 > 0.0 {
            ts.push(Intersection::new(
                scale,
                t2,
                transformed_raypos,
                normalized_transformed_raydir,
                self.0.id,
                IntersectionType::Exit,
                0,
            ));
        }
        ts
    }
    // fn inside(&self, pos: Vec3) -> bool {
    //     let transformed_pos = self.transform.inv_transform_point(pos);
    //     dot(transformed_pos, transformed_pos) <= 1.0
    // }
}

impl Primitive for Sphere {
    #[allow(clippy::many_single_char_names)]
    fn get_surface(&self, opos: Vec3, face: i64) -> Result<(Vec3, f64, f64, f64), PrimitivesError> {
        let [x, y, z] = opos;
        let v = (y + 1.0) / 2.0;
        let u = x.atan2(z);
        if face == 0 {
            Ok((self.0.surface)(face, u, v))
        } else {
            Err(PrimitivesError::InvalidFace(face))
        }
    }

    fn get_normal(&self, p: Vec3, face: i64) -> Result<Vec3, PrimitivesError> {
        if face == 0 {
            Ok(normalize(self.0.transform.transform_normal(p)))
        } else {
            Err(PrimitivesError::InvalidFace(face))
        }
    }

    fn get_transform(&self) -> &Transform {
        &self.0.transform
    }
    fn get_mut_transform(&mut self) -> &mut Transform {
        &mut self.0.transform
    }
}

static NORMALS: [Vec3; 6] = [
    [0.0, 0.0, -1.0],
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
];
static SLABS: [(i64, i64); 3] = [(3, 2), (4, 5), (1, 0)];

#[derive(Clone)]
pub struct Cube(PrimitiveCommon);

impl Cube {
    pub fn new(surface: Rc<Box<SurfaceFunction>>) -> Cube {
        Cube(PrimitiveCommon {
            transform: Default::default(),
            surface,
            id: PrimitiveId::new(),
        })
    }
}

impl IntersectRay for Cube {
    fn find_primitive(&self, id: PrimitiveId) -> Option<&Primitive> {
        if id == self.0.id {
            Some(self)
        } else {
            None
        }
    }

    fn intersect(&self, raypos: Vec3, raydir: Vec3) -> Vec<Intersection> {
        let tr = &self.0.transform;
        let transformed_raydir = tr.inv_transform_vector(raydir);
        let scale = 1.0 / length(transformed_raydir);
        let normalized_transformed_raydir = mul(transformed_raydir, scale); // normalize
        let transformed_raypos = tr.inv_transform_point(raypos);
        let eps = 1e-15;
        let mut tmin: Option<(f64, i64)> = None;
        let mut tmax: Option<(f64, i64)> = None;
        let p = sub([0.5, 0.5, 0.5], transformed_raypos);
        for i in 0..3 {
            let (mut face1, mut face2) = SLABS[i];
            let e = p[i];
            let f = normalized_transformed_raydir[i];
            if f.abs() > eps {
                let mut t1 = (e + 0.5) / f;
                let mut t2 = (e - 0.5) / f;
                if t1 > t2 {
                    mem::swap(&mut t1, &mut t2);
                    mem::swap(&mut face1, &mut face2);
                }
                tmin = match tmin {
                    Some(v) if v.0 >= t1 => tmin,
                    _ => Some((t1, face1)),
                };
                tmax = match tmax {
                    Some(v) if v.0 <= t2 => tmax,
                    _ => Some((t2, face2)),
                };
                match (tmin, tmax) {
                    (Some(tminv), Some(tmaxv)) if tminv.0 > tmaxv.0 => return vec![],
                    _ => {}
                }
                match tmax {
                    Some(b) if b.0 < 0.0 => {
                        return vec![];
                    }
                    _ => {}
                }
            } else if -e - 0.5 > 0.0 || -e + 0.5 < 0.0 {
                return vec![];
            }
        }
        let mut ts = vec![];
        match tmin {
            Some(a) if a.0 > 0.0 => {
                ts.push(Intersection::new(
                    scale,
                    a.0,
                    transformed_raypos,
                    normalized_transformed_raydir,
                    self.0.id,
                    IntersectionType::Entry,
                    a.1,
                ));
            }
            _ => {}
        }
        match tmax {
            Some(b) if b.0 > 0.0 => {
                ts.push(Intersection::new(
                    scale,
                    b.0,
                    transformed_raypos,
                    normalized_transformed_raydir,
                    self.0.id,
                    IntersectionType::Exit,
                    b.1,
                ));
            }
            _ => {}
        }
        ts
    }

    // fn inside(&self, pos: Vec3) -> bool {
    //     let [x, y, z] = self.transform.inv_transform_point(pos);
    //     0.0 <= x && x <= 1.0 && 0.0 <= y && y <= 1.0 && 0.0 <= z && z <= 1.0
    // }
}

impl Primitive for Cube {
    fn get_surface(&self, opos: Vec3, face: i64) -> Result<(Vec3, f64, f64, f64), PrimitivesError> {
        let [x, y, z] = opos;
        match face {
            0 => Ok((self.0.surface)(0, x, y)),
            1 => Ok((self.0.surface)(1, x, y)),
            2 => Ok((self.0.surface)(2, z, y)),
            3 => Ok((self.0.surface)(3, z, y)),
            4 => Ok((self.0.surface)(4, x, z)),
            5 => Ok((self.0.surface)(5, x, z)),
            _ => Err(PrimitivesError::InvalidFace(face)),
        }
    }

    fn get_normal(&self, _p: Vec3, face: i64) -> Result<Vec3, PrimitivesError> {
        if face >= 0 && (face as usize) < NORMALS.len() {
            Ok(normalize(
                self.0.transform.transform_normal(NORMALS[face as usize]),
            ))
        } else {
            Err(PrimitivesError::InvalidFace(face))
        }
    }

    fn get_transform(&self) -> &Transform {
        &self.0.transform
    }
    fn get_mut_transform(&mut self) -> &mut Transform {
        &mut self.0.transform
    }
}

#[derive(Clone)]
pub struct Cylinder(PrimitiveCommon);

impl Cylinder {
    pub fn new(surface: Rc<Box<SurfaceFunction>>) -> Cylinder {
        Cylinder(PrimitiveCommon {
            transform: Default::default(),
            surface,
            id: PrimitiveId::new(),
        })
    }

    fn solve_cyl(&self, px: f64, pz: f64, dx: f64, dz: f64) -> Option<((f64, i64), (f64, i64))> {
        // solve x ^ 2 + z ^ 2 = 1
        // (px + t * dx) ^ 2 + (pz + t * dz) ^ 2 = 1
        // a * t ^ 2 + b * t + c = 0
        // t = (-b +/- sqrt(b ^ 2 - 4 * a * c)) / 2 * a
        let a = dx * dx + dz * dz;
        let b = 2.0 * (px * dx + pz * dz);
        let c = px * px + pz * pz - 1.0;
        let sq = b * b - 4.0 * a * c;
        if sq < 0.0 {
            None
        } else {
            let root = sq.sqrt();
            let mut t1 = ((-b - root) / (2.0 * a), 0);
            let mut t2 = ((-b + root) / (2.0 * a), 0);
            if t1 > t2 {
                mem::swap(&mut t1, &mut t2);
            }
            Some((t1, t2))
        }
    }

    fn solve_plane(&self, py: f64, dy: f64) -> ((f64, i64), (f64, i64)) {
        let dinv = 1.0 / dy;
        let t1 = -py * dinv;
        let t2 = (-py + 1.0) * dinv;
        let face1 = 2i64; // bottom
        let face2 = 1i64; // top
        let mut tt1 = (t1, face1);
        let mut tt2 = (t2, face2);
        if t1 > t2 {
            mem::swap(&mut tt1, &mut tt2);
        }
        (tt1, tt2)
    }
}

impl IntersectRay for Cylinder {
    fn find_primitive(&self, id: PrimitiveId) -> Option<&Primitive> {
        if id == self.0.id {
            Some(self)
        } else {
            None
        }
    }

    fn intersect(&self, in_raypos: Vec3, in_raydir: Vec3) -> Vec<Intersection> {
        let tr = &self.0.transform;
        let mut raydir = tr.inv_transform_vector(in_raydir);
        let scale = 1.0 / length(raydir);
        raydir = mul(raydir, scale); // normalize
        let raypos = tr.inv_transform_point(in_raypos);
        let eps = 1e-7;

        let [px, py, pz] = raypos;
        let [dx, dy, dz] = raydir;
        let ts;
        if dy.abs() + eps >= 1.0 {
            // ray is parallel to the cylinder axis
            let frsd = 1.0 - px * px - pz * pz;
            if frsd < 0.0 {
                // outside cylinder
                return vec![];
            }
            ts = self.solve_plane(py, dy);
        } else if dy.abs() < eps {
            // ray is orthogonal to the cylinder axis
            // check planes
            if py < 0.0 || py > 1.0 {
                return vec![];
            }
            // check cylinder
            if let Some(res) = self.solve_cyl(px, pz, dx, dz) {
                ts = res;
            } else {
                return vec![];
            }
        } else {
            // general case
            // check cylinder
            if let Some((tc1, tc2)) = self.solve_cyl(px, pz, dx, dz) {
                // check planes
                let (tp1, tp2) = self.solve_plane(py, dy);
                // same min-max strategy as for cubes
                // Check max of mins
                let mut tmin = tp1;
                let mut tmax = tp2;
                if tc1 > tmin {
                    tmin = tc1;
                }
                if tc2 < tmax {
                    tmax = tc2;
                }
                if tmin.0 > tmax.0 {
                    return vec![];
                }
                if tmax.0 < 0.0 {
                    return vec![];
                }
                ts = (tmin, tmax);
            } else {
                return vec![];
            }
        }

        let (tmin, tmax) = ts;
        let mut it = vec![];
        if tmin.0 > 0.0 {
            it.push(Intersection::new(
                scale,
                tmin.0,
                raypos,
                raydir,
                self.0.id,
                IntersectionType::Entry,
                tmin.1,
            ));
        }
        if tmax.0 > 0.0 {
            it.push(Intersection::new(
                scale,
                tmax.0,
                raypos,
                raydir,
                self.0.id,
                IntersectionType::Exit,
                tmax.1,
            ));
        }
        it
    }

    // fn inside(&self, pos: Vec3) -> bool {
    //     let [x, y, z] = self.0.transform.inv_transform_point(pos);
    //     (x * x + z * z) <= 1.0 && 0.0 <= y && y <= 1.0
    // }
}

impl Primitive for Cylinder {
    fn get_surface(&self, opos: Vec3, face: i64) -> Result<(Vec3, f64, f64, f64), PrimitivesError> {
        let [x, y, z] = opos;
        match face {
            0 => Ok((self.0.surface)(0, x.atan2(z), y)),
            1 => Ok((self.0.surface)(1, (x + 1.0) / 2.0, (z + 1.0) / 2.0)),
            2 => Ok((self.0.surface)(2, (x + 1.0) / 2.0, (z + 1.0) / 2.0)),
            face => Err(PrimitivesError::InvalidFace(face)),
        }
    }
    fn get_normal(&self, p: Vec3, face: i64) -> Result<Vec3, PrimitivesError> {
        let n = match face {
            0 => Ok([p[0], 0.0, p[2]]),
            1 => Ok([0.0, 1.0, 0.0]),
            2 => Ok([0.0, -1.0, 0.0]),
            _ => Err(PrimitivesError::InvalidFace(face)),
        };
        Ok(normalize(self.0.transform.transform_normal(n?)))
    }

    fn get_transform(&self) -> &Transform {
        &self.0.transform
    }
    fn get_mut_transform(&mut self) -> &mut Transform {
        &mut self.0.transform
    }
}

#[derive(Clone)]
pub struct Cone(PrimitiveCommon);

impl Cone {
    pub fn new(surface: Rc<Box<SurfaceFunction>>) -> Cone {
        Cone(PrimitiveCommon {
            transform: Default::default(),
            surface,
            id: PrimitiveId::new(),
        })
    }

    fn solve_cone(
        &self,
        px: f64,
        py: f64,
        pz: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> Option<((f64, i64), (f64, i64))> {
        // solve x ^ 2 + z ^ 2 = y ^ 2
        // (px + t * dx) ^ 2 + (pz + t * dz) ^ 2 = (py + t * dy) ^ 2
        // a * t ^ 2 + b * t + c = 0
        // t = (-b +/- sqrt(b ^ 2 - 4 * a * c)) / 2 * a
        let a = dx * dx + dz * dz - dy * dy;
        let b = 2.0 * (px * dx + pz * dz - py * dy);
        let c = px * px + pz * pz - py * py;
        let sq = b * b - 4.0 * a * c;
        if sq < 0.0 {
            None
        } else {
            let root = sq.sqrt();
            let mut t1 = ((-b - root) / (2.0 * a), 0);
            let mut t2 = ((-b + root) / (2.0 * a), 0);
            if t1 > t2 {
                mem::swap(&mut t1, &mut t2);
            }
            Some((t1, t2))
        }
    }
}

impl IntersectRay for Cone {
    fn find_primitive(&self, id: PrimitiveId) -> Option<&Primitive> {
        if id == self.0.id {
            Some(self)
        } else {
            None
        }
    }

    fn intersect(&self, in_raypos: Vec3, in_raydir: Vec3) -> Vec<Intersection> {
        let tr = &self.0.transform;
        let mut raydir = tr.inv_transform_vector(in_raydir);
        let scale = 1.0 / length(raydir);
        raydir = mul(raydir, scale); // normalize
        let raypos = tr.inv_transform_point(in_raypos);

        let [px, py, pz] = raypos;
        let [dx, dy, dz] = raydir;
        let tsc = self.solve_cone(px, py, pz, dx, dy, dz);
        if let Some((tcmin, tcmax)) = tsc {
            let mut ts = vec![];
            let cminy = py + tcmin.0 * dy;
            let cmaxy = py + tcmax.0 * dy;
            if 0.0 <= cminy && cminy <= 1.0 {
                ts.push(tcmin)
            }
            if 0.0 <= cmaxy && cmaxy <= 1.0 {
                ts.push(tcmax)
            }
            if ts.is_empty() {
                return vec![];
            }
            if ts.len() == 2 {
                let mut tr = vec![];
                if ts[0].0 > 0.0 {
                    tr.push(Intersection::new(
                        scale,
                        ts[0].0,
                        raypos,
                        raydir,
                        self.0.id,
                        IntersectionType::Entry,
                        ts[0].1,
                    ));
                }
                if ts[1].0 > 0.0 {
                    tr.push(Intersection::new(
                        scale,
                        ts[1].0,
                        raypos,
                        raydir,
                        self.0.id,
                        IntersectionType::Exit,
                        ts[1].1,
                    ));
                }
                return tr;
            }
            // check plane
            // since we know there is only one intersection with the cone,
            // there must be an intersection in the base
            let tp = (-py + 1.0) / dy;
            ts.push((tp, 1));
            ts.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Less));
            let mut tr = vec![];
            if ts[0].0 > 0.0 {
                tr.push(Intersection::new(
                    scale,
                    ts[0].0,
                    raypos,
                    raydir,
                    self.0.id,
                    IntersectionType::Entry,
                    ts[0].1,
                ));
            }
            if ts[1].0 > 0.0 {
                tr.push(Intersection::new(
                    scale,
                    ts[1].0,
                    raypos,
                    raydir,
                    self.0.id,
                    IntersectionType::Exit,
                    ts[1].1,
                ));
            }
            tr
        } else {
            vec![]
        }
    }

    // fn inside(&self, pos: Vec3) -> bool {
    //     let [x, y, z] = self.0.transform.inv_transform_point(pos);
    //     let p = x * x - y * y + z * z;
    //     p <= 0.0 && 0.0 <= y && y <= 1.0
    // }
}

impl Primitive for Cone {
    fn get_surface(&self, opos: Vec3, face: i64) -> Result<(Vec3, f64, f64, f64), PrimitivesError> {
        let [x, y, z] = opos;
        match face {
            0 => Ok((self.0.surface)(0, x.atan2(z), y)),
            1 => Ok((self.0.surface)(1, (x + 1.0) / 2.0, (z + 1.0) / 2.0)),
            face => Err(PrimitivesError::InvalidFace(face)),
        }
    }

    fn get_normal(&self, p: Vec3, face: i64) -> Result<Vec3, PrimitivesError> {
        let n = match face {
            0 => Ok([p[0], -p[1], p[2]]),
            1 => Ok([0.0, 1.0, 0.0]),
            _ => Err(PrimitivesError::InvalidFace(face)),
        };
        Ok(normalize(self.0.transform.transform_normal(n?)))
    }

    fn get_transform(&self) -> &Transform {
        &self.0.transform
    }
    fn get_mut_transform(&mut self) -> &mut Transform {
        &mut self.0.transform
    }
}

#[derive(Clone)]
pub struct Plane(PrimitiveCommon);

impl Plane {
    pub fn new(surface: Rc<Box<SurfaceFunction>>) -> Self {
        Plane(PrimitiveCommon {
            transform: Default::default(),
            surface,
            id: PrimitiveId::new(),
        })
    }
}

impl IntersectRay for Plane {
    fn find_primitive(&self, id: PrimitiveId) -> Option<&Primitive> {
        if id == self.0.id {
            Some(self)
        } else {
            None
        }
    }

    fn intersect(&self, in_raypos: Vec3, in_raydir: Vec3) -> Vec<Intersection> {
        let np = [0.0, 1.0, 0.0];
        let tr = &self.0.transform;
        let mut raydir = tr.inv_transform_vector(in_raydir);
        let scale = 1.0 / length(raydir);
        raydir = mul(raydir, scale);
        let raypos = tr.inv_transform_point(in_raypos);
        let denom = dot(np, raydir);
        if denom.abs() < 1e-7 {
            return vec![];
        }
        let t = -dot(np, raypos) / denom;
        if t < 0.0 {
            return vec![];
        }
        if denom > 0.0 {
            return vec![Intersection::new(
                scale,
                t,
                raypos,
                raydir,
                self.0.id,
                IntersectionType::Exit,
                0,
            )];
        } else {
            return vec![Intersection::new(
                scale,
                t,
                raypos,
                raydir,
                self.0.id,
                IntersectionType::Entry,
                0,
            )];
        }
    }

    // fn inside(&self, pos: Vec3) -> bool {
    //     self.0.transform.inv_transform_py(pos) <= 0.0
    // }
}

impl Primitive for Plane {
    fn get_surface(&self, opos: Vec3, face: i64) -> Result<(Vec3, f64, f64, f64), PrimitivesError> {
        let [x, _y, z] = opos;
        if face == 0 {
            Ok((self.0.surface)(0, x, z))
        } else {
            Err(PrimitivesError::InvalidFace(face))
        }
    }

    fn get_normal(&self, _p: Vec3, face: i64) -> Result<Vec3, PrimitivesError> {
        if face == 0 {
            Ok(normalize(
                self.0.transform.transform_normal([0.0, 1.0, 0.0]),
            ))
        } else {
            Err(PrimitivesError::InvalidFace(face))
        }
    }

    fn get_transform(&self) -> &Transform {
        &self.0.transform
    }
    fn get_mut_transform(&mut self) -> &mut Transform {
        &mut self.0.transform
    }
}

#[test]
fn test_intersection() {
    let mut i = Intersection::new(
        2.0,
        3.0,
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        PrimitiveId::new(),
        IntersectionType::Entry,
        0,
    );
    assert!(i.distance == 6.0);
    assert!(i.t == IntersectionType::Entry);
    i.switch(IntersectionType::Exit);
    assert!(i.t == IntersectionType::Exit);
}

#[test]
fn test_intersection_sphere() {
    let mut obj = Box::new(Sphere::new(Rc::new(Box::new(|_face, _u, _v| {
        ([1.0, 0.0, 0.0], 0.9, 0.9, 0.9)
    }))));
    obj.translate(0.0, 0.0, 5.0);
    let mut intersections = obj.intersect([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    for i in intersections.iter_mut() {
        println!("Intersection type: {:?}, distance: {:?}", i.t, i.distance);
    }
}
