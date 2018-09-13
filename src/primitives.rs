use vecmath::{Vec3, add, dot, mul, neg, length, normalize, sub};
use transform::Transform;
use std::rc::Rc;
use std::cmp::Ordering;
use std::mem;
use std::iter;
use std::sync::atomic::{self, AtomicUsize};

static NODE_COUNTER: AtomicUsize = atomic::ATOMIC_USIZE_INIT;

#[derive(Debug, PartialEq, Copy)]
pub struct NodeId(usize);

impl NodeId {
    fn new() -> Self {
        NodeId(NODE_COUNTER.fetch_add(1, atomic::Ordering::SeqCst))
    }
}

impl Clone for NodeId {
    fn clone(&self) -> NodeId {
        NodeId::new()
    }
}

pub trait Node: NodeClone {
    fn name(&self) -> &str;
    fn id(&self) -> NodeId;
    fn find_node(&self, id: NodeId) -> Option<&Node>;
    fn intersect(&self, raypos: Vec3, raydir: Vec3) -> Vec<Intersection>;
    fn inside(&self, pos: Vec3) -> bool;
    fn get_surface(&self, opos: Vec3, face: i64) -> (Vec3, f64, f64, f64);
    fn translate(&mut self, tx: f64, ty: f64, tz: f64);
    fn scale(&mut self, sx: f64, sy: f64, sz: f64);
    fn uscale(&mut self, s: f64);
    fn rotatex(&mut self, d: f64);
    fn rotatey(&mut self, d: f64);
    fn rotatez(&mut self, d: f64);
    fn transform_point(&self, p: Vec3) -> Vec3;
    fn get_normal(&self, p: Vec3, face: i64) -> Vec3;
}

pub trait NodeClone {
    fn clone_box(&self) -> Box<Node>;
}

impl<T> NodeClone for T
    where T: 'static + Node + Clone
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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum IntersectionType {
    Entry,
    Exit
}

#[derive(Copy, Clone)]
pub struct Intersection {
    scale: f64,
    odistance: f64,
    pub distance: f64,
    rp: Vec3,
    rd: Vec3,
    pub primitive_id: NodeId,
    pub t: IntersectionType,
    pub face: i64  // Todo: maybe use a type instea
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
        primitive_id: NodeId,
        t: IntersectionType,
        face: i64
    ) -> Intersection {
        Intersection {
            scale: scale,
            odistance: odistance,
            distance: scale * odistance,
            rp: rp,
            rd: rd,
            primitive_id: primitive_id,
            t: t,
            face: face
        }
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
    name: &'static str,
    id: NodeId
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
            obj1: obj1,
            obj2: obj2,
            rule: &union,
            name: "union",
            id: NodeId::new()
        }
    }
    pub fn make_intersect(obj1: Box<Node>, obj2: Box<Node>) -> Operator {
        Operator {
            obj1: obj1,
            obj2: obj2,
            rule: &intersect,
            name: "intersect",
            id: NodeId::new()
        }
    }
    pub fn make_difference(obj1: Box<Node>, obj2: Box<Node>) -> Operator {
        Operator {
            obj1: obj1,
            obj2: obj2,
            rule: &difference,
            name: "difference",
            id: NodeId::new()
        }
    }
}

impl Node for Operator {
    fn name(&self) -> &str {
        self.name
    }
    fn id(&self) -> NodeId {
        self.id
    }
    fn find_node(&self, id: NodeId) -> Option<&Node> {
        if let Some(n) = self.obj1.find_node(id) {
            Some(n)
        } else if let Some(n) = self.obj2.find_node(id) {
            Some(n)
        } else {
            None
        }
    }
    fn inside(&self, pos: Vec3) -> bool {
        (self.rule)(self.obj1.inside(pos), self.obj2.inside(pos))
    }
    fn get_surface(&self, _opos: Vec3, _face: i64) -> (Vec3, f64, f64, f64) {
        panic!("this should not happen");
    }
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

    fn transform_point(&self, _p: Vec3) -> Vec3 {
        panic!("this should not happen");
    }

    fn get_normal(&self, _p: Vec3, _face: i64) -> Vec3 {
        panic!("this should not happen");
    }

    fn intersect(&self, raypos: Vec3, raydir: Vec3) -> Vec<Intersection> {
        let mut inside1 = 0;
        let mut inside2 = 0;
        if self.obj1.inside(raypos) {
            inside1 = 1;
        }
        if self.obj2.inside(raypos) {
            inside2 = 1;
        }
        let mut inside = (self.rule)(inside1 > 0, inside2 > 0);
        let mut obj1i = self.obj1.intersect(raypos, raydir);
        let mut obj2i = self.obj2.intersect(raypos, raydir);

        let mut intersections: Vec<(&mut Intersection, i32)> = obj1i.iter_mut()
            .zip(iter::repeat(1))
            .chain(obj2i.iter_mut()
                   .zip(iter::repeat(2)))
            .collect();
        intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut res = Vec::<Intersection>::new();
        let mut prevt = 0.0;
        for (ref mut i, obj) in intersections.iter_mut() {
            match i.t {
                IntersectionType::Entry => {
                    if *obj == 1 {
                        inside1 = inside1 + 1;
                    } else {
                        inside2 = inside2 + 1;
                    }
                },
                IntersectionType::Exit => {
                    if *obj == 1 {
                        inside1 = inside1 - 1;
                    } else {
                        inside2 = inside2 - 1;
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
                    i.switch(IntersectionType::Exit);
                    res.push(**i);
                }
            }
            if !inside && newinside {
                i.switch(IntersectionType::Entry);
                prevt = i.distance;
                res.push(**i);
                inside = newinside;
            }
        }

        res
    }
}

#[derive(Clone)]
pub struct Sphere {
    transform: Transform,
    surface: Rc<Box<Fn(i64, f64, f64) -> (Vec3, f64, f64, f64)>>,
    id: NodeId
}

impl Sphere {
    pub fn new(surface: Rc<Box<Fn(i64, f64, f64) -> (Vec3, f64, f64, f64)>>) -> Sphere {
        Sphere { transform: Default::default(), surface: surface, id: NodeId::new() }
    }
}

impl Node for Sphere {
    fn name(&self) -> &str {
        return "sphere";
    }

    fn id(&self) -> NodeId {
        self.id
    }

    fn find_node(&self, id: NodeId) -> Option<&Node> {
        if id == self.id {
            Some(self)
        } else {
            None
        }
    }

    fn intersect(&self, raypos: Vec3, raydir: Vec3) -> Vec<Intersection> {
        let tr = &self.transform;
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
            ts.push(Intersection::new(scale, t1, transformed_raypos, normalized_transformed_raydir, self.id(), IntersectionType::Entry, 0));
        }
        if t2 > 0.0 {
            ts.push(Intersection::new(scale, t2, transformed_raypos, normalized_transformed_raydir, self.id(), IntersectionType::Exit, 0));
        }
        ts
    }
    fn inside(&self, pos: Vec3) -> bool {
        let transformed_pos = self.transform.inv_transform_point(pos);
        dot(transformed_pos, transformed_pos) <= 1.0
    }
    fn get_surface(&self, opos: Vec3, face: i64) -> (Vec3, f64, f64, f64) {
        let [x, y, z] = opos;
        let v = (y + 1.0) / 2.0;
        let u = x.atan2(z);
        (self.surface)(face, u, v)
    }
    fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
        self.transform.translate(tx, ty, tz);
    }
    fn scale(&mut self, sx: f64, sy: f64, sz: f64) {
        self.transform.scale(sx, sy, sz);
    }
    fn uscale(&mut self, s: f64) {
        self.transform.uscale(s);
    }
    fn rotatex(&mut self, d: f64) {
        self.transform.rotatex(d);
    }
    fn rotatey(&mut self, d: f64) {
        self.transform.rotatey(d);
    }
    fn rotatez(&mut self, d: f64) {
        self.transform.rotatez(d);
    }

    fn transform_point(&self, p: Vec3) -> Vec3 {
        self.transform.transform_point(p)
    }

    fn get_normal(&self, p: Vec3, _face: i64) -> Vec3 {
        normalize(self.transform.transform_normal(p))
    }
}

static NORMALS : [Vec3;6] = [[0.0, 0.0, -1.0],
                             [0.0, 0.0, 1.0],
                             [-1.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, -1.0, 0.0]];
static SLABS : [(i64, i64);3] = [(3, 2),
                                 (4, 5),
                                 (1, 0)];

#[derive(Clone)]
pub struct Cube {
    transform: Transform,
    surface: Rc<Box<Fn(i64, f64, f64) -> (Vec3, f64, f64, f64)>>,
    id: NodeId
}

impl Cube {
    pub fn new(surface: Rc<Box<Fn(i64, f64, f64) -> (Vec3, f64, f64, f64)>>) -> Cube {
        Cube { transform: Default::default(), surface: surface, id: NodeId::new() }
    }
}

impl Node for Cube {
    fn name(&self) -> &str {
        return "cube";
    }

    fn id(&self) -> NodeId {
        self.id
    }

    fn find_node(&self, id: NodeId) -> Option<&Node> {
        if id == self.id {
            Some(self)
        } else {
            None
        }
    }

    fn intersect(&self, raypos: Vec3, raydir: Vec3) -> Vec<Intersection> {
        let tr = &self.transform;
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
                if tmin.is_none() || t1 > tmin.unwrap().0 {
                    tmin = Some((t1, face1));
                }
                if tmax.is_none() || t2 < tmax.unwrap().0 {
                    tmax = Some((t2, face2));
                }
                if tmin.unwrap().0 > tmax.unwrap().0 {
                    return vec![];
                }
                if tmax.unwrap().0 < 0.0 {
                    return vec![];
                }
            } else if -e - 0.5 > 0.0 || -e + 0.5 < 0.0 {
                return vec![];
            }
        }
        let mut ts = vec![];
        if tmin.unwrap().0 > 0.0 {
            ts.push(Intersection::new(scale, tmin.unwrap().0, transformed_raypos, normalized_transformed_raydir, self.id(), IntersectionType::Entry, tmin.unwrap().1));
        }
        if tmax.unwrap().0 > 0.0 {
            ts.push(Intersection::new(scale, tmax.unwrap().0, transformed_raypos, normalized_transformed_raydir, self.id(), IntersectionType::Exit, tmax.unwrap().1));
        }
        ts
    }

    fn inside(&self, pos: Vec3) -> bool {
        let [x, y, z] = self.transform.inv_transform_point(pos);
        0.0 <= x && x <= 1.0 && 0.0 <= y && y <= 1.0 && 0.0 <= z && z <= 1.0
    }

    fn get_surface(&self, opos: Vec3, face: i64) -> (Vec3, f64, f64, f64) {
        let [x, y, z] = opos;
        match face {
            0 => (self.surface)(0, x, y),
            1 => (self.surface)(1, x, y),
            2 => (self.surface)(2, z, y),
            3 => (self.surface)(3, z, y),
            4 => (self.surface)(4, x, z),
            5 => (self.surface)(5, x, z),
            _ => panic!("unexpected face")
        }
    }
    fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
        self.transform.translate(tx, ty, tz);
    }
    fn scale(&mut self, sx: f64, sy: f64, sz: f64) {
        self.transform.scale(sx, sy, sz);
    }
    fn uscale(&mut self, s: f64) {
        self.transform.uscale(s);
    }
    fn rotatex(&mut self, d: f64) {
        self.transform.rotatex(d);
    }
    fn rotatey(&mut self, d: f64) {
        self.transform.rotatey(d);
    }
    fn rotatez(&mut self, d: f64) {
        self.transform.rotatez(d);
    }

    fn transform_point(&self, p: Vec3) -> Vec3 {
        self.transform.transform_point(p)
    }

    fn get_normal(&self, _p: Vec3, face: i64) -> Vec3 {
        normalize(self.transform.transform_normal(NORMALS[face as usize]))
    }
}

#[derive(Clone)]
pub struct Cylinder {
    transform: Transform,
    surface: Rc<Box<Fn(i64, f64, f64) -> (Vec3, f64, f64, f64)>>,
    id: NodeId
}

impl Cylinder {
    pub fn new(surface: Rc<Box<Fn(i64, f64, f64) -> (Vec3, f64, f64, f64)>>) -> Cylinder {
        Cylinder { transform: Default::default(), surface: surface, id: NodeId::new() }
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

impl Node for Cylinder {
    fn name(&self) -> &str {
        return "cylinder";
    }

    fn id(&self) -> NodeId {
        self.id
    }

    fn find_node(&self, id: NodeId) -> Option<&Node> {
        if id == self.id {
            Some(self)
        } else {
            None
        }
    }


    fn intersect(&self, in_raypos: Vec3, in_raydir: Vec3) -> Vec<Intersection> {
        let tr = &self.transform;
        let mut raydir = tr.inv_transform_vector(in_raydir);
        let scale = 1.0 / length(raydir);
        raydir = mul(raydir, scale); // normalize
        let raypos = tr.inv_transform_point(in_raypos);
        let eps = 1e-7;

        let [px, py, pz] = raypos;
        let [dx, dy, dz] = raydir;
        let mut ts = ((0.0, 0), (0.0, 0));
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
            it.push(Intersection::new(scale, tmin.0, raypos, raydir, self.id(), IntersectionType::Entry, tmin.1));
        }
        if tmax.0 > 0.0 {
            it.push(Intersection::new(scale, tmax.0, raypos, raydir, self.id(), IntersectionType::Exit, tmax.1));
        }
        it
    }

    fn inside(&self, pos: Vec3) -> bool {
        let [x, y, z] = self.transform.inv_transform_point(pos);
        (x * x + z * z) <= 1.0 && 0.0 <= y && y <= 1.0
    }

    fn get_surface(&self, opos: Vec3, face: i64) -> (Vec3, f64, f64, f64) {
        let [x, y, z] = opos;
        match face {
            0 => (self.surface)(0, x.atan2(z), y),
            1 => (self.surface)(1, (x + 1.0) / 2.0, (z + 1.0) / 2.0),
            2 => (self.surface)(2, (x + 1.0) / 2.0, (z + 1.0) / 2.0),
            _ => panic!("invalid face")
        }
    }

    fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
        self.transform.translate(tx, ty, tz);
    }
    fn scale(&mut self, sx: f64, sy: f64, sz: f64) {
        self.transform.scale(sx, sy, sz);
    }
    fn uscale(&mut self, s: f64) {
        self.transform.uscale(s);
    }
    fn rotatex(&mut self, d: f64) {
        self.transform.rotatex(d);
    }
    fn rotatey(&mut self, d: f64) {
        self.transform.rotatey(d);
    }
    fn rotatez(&mut self, d: f64) {
        self.transform.rotatez(d);
    }

    fn transform_point(&self, p: Vec3) -> Vec3 {
        self.transform.transform_point(p)
    }

    fn get_normal(&self, p: Vec3, face: i64) -> Vec3 {
        let n = match face {
            0 => [p[0], 0.0, p[2]],
            1 => [0.0, 1.0, 0.0],
            2 => [0.0, -1.0, 0.0],
            _ => panic!("invalid face")
        };
        normalize(self.transform.transform_normal(n))
    }
}

#[derive(Clone)]
pub struct Cone {
    transform: Transform,
    surface: Rc<Box<Fn(i64, f64, f64) -> (Vec3, f64, f64, f64)>>,
    id: NodeId
}

impl Cone {
    pub fn new(surface: Rc<Box<Fn(i64, f64, f64) -> (Vec3, f64, f64, f64)>>) -> Cone {
        Cone { transform: Default::default(), surface: surface, id: NodeId::new() }
    }

    fn solve_cone(&self, px: f64, py: f64, pz: f64, dx: f64, dy: f64, dz: f64) -> Option<((f64, i64), (f64, i64))> {
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

impl Node for Cone {
    fn name(&self) -> &str {
        return "cone";
    }

    fn id(&self) -> NodeId {
        self.id
    }

    fn find_node(&self, id: NodeId) -> Option<&Node> {
        if id == self.id {
            Some(self)
        } else {
            None
        }
    }


    fn intersect(&self, in_raypos: Vec3, in_raydir: Vec3) -> Vec<Intersection> {
        let tr = &self.transform;
        let mut raydir = tr.inv_transform_vector(in_raydir);
        let scale = 1.0 / length(raydir);
        raydir = mul(raydir, scale); // normalize
        let raypos = tr.inv_transform_point(in_raypos);
        let eps = 1e-7;

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
            if ts.len() == 0 {
                return vec![];
            }
            if ts.len() == 2 {
                let mut tr = vec![];
                if ts[0].0 > 0.0 {
                    tr.push(Intersection::new(scale, ts[0].0, raypos, raydir, self.id(), IntersectionType::Entry, ts[0].1));
                }
                if ts[1].0 > 0.0 {
                    tr.push(Intersection::new(scale, ts[1].0, raypos, raydir, self.id(), IntersectionType::Exit, ts[1].1));
                }
                return tr;
            }
            // check plane
            // since we know there is only one intersection with the cone,
            // there must be an intersection in the base
            let tp = (-py + 1.0) / dy;
            ts.push((tp, 1));
            ts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut tr = vec![];
            if ts[0].0 > 0.0 {
                tr.push(Intersection::new(scale, ts[0].0, raypos, raydir, self.id(), IntersectionType::Entry, ts[0].1));
            }
            if ts[1].0 > 0.0 {
                tr.push(Intersection::new(scale, ts[1].0, raypos, raydir, self.id(), IntersectionType::Exit, ts[1].1));
            }
            tr
        } else {
            vec![]
        }
    }

    fn inside(&self, pos: Vec3) -> bool {
        let [x, y, z] = self.transform.inv_transform_point(pos);
        let p = x * x - y * y + z * z;
        p <= 0.0 && 0.0 <= y && y <= 1.0
    }

    fn get_surface(&self, opos: Vec3, face: i64) -> (Vec3, f64, f64, f64) {
        let [x, y, z] = opos;
        match face {
            0 => (self.surface)(0, x.atan2(z), y),
            1 => (self.surface)(1, (x + 1.0) / 2.0, (z + 1.0) / 2.0),
            _ => panic!("invalid face")
        }
    }

    fn get_normal(&self, p: Vec3, face: i64) -> Vec3 {
        let n = match face {
            0 => [p[0], -p[1], p[2]],
            1 => [0.0, 1.0, 0.0],
            _ => panic!("invalid face")
        };
        normalize(self.transform.transform_normal(n))
    }
    fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
        self.transform.translate(tx, ty, tz);
    }
    fn scale(&mut self, sx: f64, sy: f64, sz: f64) {
        self.transform.scale(sx, sy, sz);
    }
    fn uscale(&mut self, s: f64) {
        self.transform.uscale(s);
    }
    fn rotatex(&mut self, d: f64) {
        self.transform.rotatex(d);
    }
    fn rotatey(&mut self, d: f64) {
        self.transform.rotatey(d);
    }
    fn rotatez(&mut self, d: f64) {
        self.transform.rotatez(d);
    }

    fn transform_point(&self, p: Vec3) -> Vec3 {
        self.transform.transform_point(p)
    }
}

#[derive(Clone)]
pub struct Plane {
    transform: Transform,
    surface: Rc<Box<Fn(i64, f64, f64) -> (Vec3, f64, f64, f64)>>,
    id: NodeId
}

impl Plane {
    pub fn new(surface: Rc<Box<Fn(i64, f64, f64) -> (Vec3, f64, f64, f64)>>) -> Self {
        Plane { transform: Default::default(), surface: surface, id: NodeId::new() }
    }
}

impl Node for Plane {
    fn name(&self) -> &str {
        return "plane";
    }

    fn id(&self) -> NodeId {
        self.id
    }

    fn find_node(&self, id: NodeId) -> Option<&Node> {
        if id == self.id {
            Some(self)
        } else {
            None
        }
    }

    fn intersect(&self, raypos: Vec3, raydir: Vec3) -> Vec<Intersection> {
        let np = [0.0, 1.0, 0.0];
        let tr = &self.transform;
        let transformed_raydir = tr.inv_transform_vector(raydir);
        let scale = 1.0 / length(transformed_raydir);
        let normalized_transformed_raydir = mul(transformed_raydir, scale);
        let transformed_raypos = tr.inv_transform_point(raypos);
        let denom = dot(np, normalized_transformed_raydir);
        if denom.abs() < 1e-7 {
            return vec![];
        }
        let t = -dot(np, transformed_raypos) / denom;
        if t < 0.0 {
            return vec![];
        }
        if denom > 0.0 {
            return vec![Intersection::new(scale, t, transformed_raypos, normalized_transformed_raydir, self.id, IntersectionType::Exit, 0)];
        } else {
            return vec![Intersection::new(scale, t, transformed_raypos, normalized_transformed_raydir, self.id, IntersectionType::Entry, 0)];
        }
    }

    fn inside(&self, pos: Vec3) -> bool {
        self.transform.inv_transform_py(pos) <= 0.0
    }

    fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
        self.transform.translate(tx, ty, tz);
    }
    fn scale(&mut self, sx: f64, sy: f64, sz: f64) {
        self.transform.scale(sx, sy, sz);
    }
    fn uscale(&mut self, s: f64) {
        self.transform.uscale(s);
    }
    fn rotatex(&mut self, d: f64) {
        self.transform.rotatex(d);
    }
    fn rotatey(&mut self, d: f64) {
        self.transform.rotatey(d);
    }
    fn rotatez(&mut self, d: f64) {
        self.transform.rotatez(d);
    }

    fn transform_point(&self, p: Vec3) -> Vec3 {
        self.transform.transform_point(p)
    }

    fn get_surface(&self, opos: Vec3, _face: i64) -> (Vec3, f64, f64, f64) {
        let [x, _y, z] = opos;
        (self.surface)(0, x, z)
    }

    fn get_normal(&self, _p: Vec3, _face: i64) -> Vec3 {
        normalize(self.transform.transform_normal([0.0, 1.0, 0.0]))
    }
}

#[test]
fn test_intersection() {
    let mut i = Intersection::new(
        2.0,
        3.0,
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        NodeId::new(),
        IntersectionType::Entry,
        0);
    assert!(i.distance == 6.0);
    assert!(i.t == IntersectionType::Entry);
    i.switch(IntersectionType::Exit);
    assert!(i.t == IntersectionType::Exit);
}

#[test]
fn test_intersection_sphere() {
    let mut obj = Box::new(Sphere::new(Rc::new(Box::new(|_face, _u, _v| ([1.0, 0.0, 0.0], 0.9, 0.9, 0.9)))));
    obj.translate(0.0, 0.0, 5.0);
    let mut intersections = obj.intersect([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]);
    for i in intersections.iter_mut() {
        println!("Intersection type: {:?}, distance: {:?}", i.t, i.distance);
    }
}
