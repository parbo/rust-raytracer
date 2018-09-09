use vecmath::{Vec3, add, dot, mul, neg, length};
use transform::Transform;
use std::rc::Rc;
use std::cmp::Ordering;
use std::mem;

pub trait Node: NodeClone {
    fn name(&self) -> &str;
    fn intersect(&self, raypos: Vec3, raydir: Vec3) -> Vec<Intersection>;
    fn inside(&self, pos: Vec3) -> bool;
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

// pub struct Operator {
//     obj1: Box<Node>,
//     obj2: Box<Node>,
//     rule: Fn(bool, bool) -> bool,
// }

#[derive(Clone)]
pub struct Sphere {
    transform: Transform,
    surface: Rc<Box<Fn(i64, f64, f64) -> (Vec3, f64, f64, f64)>>
}

impl Sphere {
    pub fn new(surface: Rc<Box<Fn(i64, f64, f64) -> (Vec3, f64, f64, f64)>>) -> Sphere {
        Sphere { transform: Default::default(), surface: surface }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IntersectionType {
    Entry,
    Exit
}

#[derive(Clone)]
pub struct Intersection {
    scale: f64,
    odistance: f64,
    distance: f64,
    rp: Vec3,
    rd: Vec3,
    primitive_transform: Transform,
    t: IntersectionType,
    face: i64,  // Todo: maybe use a type instea
    wpos: Option<Vec3>,
    opos: Option<Vec3>,
    normal: Option<Vec3>
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
        primitive_transform: Transform,
        t: IntersectionType,
        face: i64
    ) -> Intersection {
        Intersection {
            scale: scale,
            odistance: odistance,
            distance: scale * odistance,
            rp: rp,
            rd: rd,
            primitive_transform: primitive_transform,
            t: t,
            face: face,
            wpos: None,
            opos: None,
            normal: None
        }
    }

    fn switch(&mut self, t: IntersectionType) {
        if self.t != t {
            self.t = t;
            self.normal = Some(neg(self.get_normal()));
        }
    }

    fn get_normal(&mut self) -> Vec3 {
        if let Some(normal) = self.normal {
            normal
        } else {
            // TODO: actually calculate it
            let normal = [1.0, 2.0, 3.0];
            self.normal = Some(normal);
            normal
        }
    }

    fn get_opos(&mut self) -> Vec3 {
        if let Some(opos) = self.opos {
            opos
        } else {
            let opos = add(self.rp, mul(self.rd, self.odistance));
            self.opos = Some(opos);
            opos
        }
    }

    fn get_wpos(&mut self) -> Vec3 {
        if let Some(wpos) = self.wpos {
            wpos
        } else {
            let opos = self.get_opos();
            let wpos = self.primitive_transform.transform_point(opos);
            self.wpos = Some(wpos);
            wpos
        }
    }
}

// class Node(object):
//     def intersect(self, raypos, raydir):
//         return []

// class Operator(Node):
//     def __init__(self, obj1, obj2):
//         self.obj1 = obj1
//         self.obj2 = obj2

//     def translate(self, tx, ty, tz):
//         self.obj1.translate(tx, ty, tz)
//         self.obj2.translate(tx, ty, tz)

//     def scale(self, sx, sy, sz):
//         self.obj1.scale(sx, sy, sz)
//         self.obj2.scale(sx, sy, sz)

//     def uscale(self, s):
//         self.obj1.uscale(s)
//         self.obj2.uscale(s)

//     def rotatex(self, d):
//         self.obj1.rotatex(d)
//         self.obj2.rotatex(d)

//     def rotatey(self, d):
//         self.obj1.rotatey(d)
//         self.obj2.rotatey(d)

//     def rotatez(self, d):
//         self.obj1.rotatez(d)
//         self.obj2.rotatez(d)

//     def inside(self, pos):
//         return self.rule(self.obj1.inside(pos), self.obj2.inside(pos))

//     def intersect(self, raypos, raydir):
//         inside1 = 0
//         inside2 = 0
//         if self.obj1.inside(raypos):
//             inside1 = 1
//         if self.obj2.inside(raypos):
//             inside2 = 1
//         inside = self.rule(inside1 > 0, inside2 > 0)
//         #print inside, inside1, inside2, self.obj1, self.obj2
//         obj1i = self.obj1.intersect(raypos, raydir)
//         obj2i = self.obj2.intersect(raypos, raydir)
//         intersections = sorted([(i, 1) for i in obj1i] +
//                                [(i, 2) for i in obj2i])
//         res = []
//         prevt = 0.0
//         for i, obj in intersections:
//             if i.t == Intersection.ENTRY:
//                 if obj == 1:
//                     inside1 += 1
//                 else:
//                     inside2 += 1
//             elif i.t == Intersection.EXIT:
//                 if obj == 1:
//                     inside1 -= 1
//                 else:
//                     inside2 -= 1

//             newinside = self.rule(inside1 > 0, inside2 > 0)
// #            print i, inside1, inside2, inside, newinside
//             if inside and not newinside:
//                 if (i.distance - prevt) < 1e-10:
//                     # remove infinitesimal intersections
//                     # to avoid problem with difference of touching surfaces
//                     #print i, res[-1]
//                     res.pop()
//                 else:
//                     i.switch(Intersection.EXIT)
//                     res.append(i)
//             if not inside and newinside:
//                 i.switch(Intersection.ENTRY)
//                 res.append(i)
//                 prevt = i.distance
//             inside = newinside

// #        for r in res:
// #            print "r", r
//         return res

// class Union(Operator):
//     def rule(self, a, b):
//         return a or b

// class Intersect(Operator):
//     def rule(self, a, b):
//         return a and b

// class Difference(Operator):
//     def rule(self, a, b):
//         return a and not b

// class Primitive(Node):
//     def __init__(self, surface):
//         self.surface = surface
//         self.transform = Transform()

//     def translate(self, tx, ty, tz):
//         self.transform.translate(tx, ty, tz)

//     def scale(self, sx, sy, sz):
//         self.transform.scale(sx, sy, sz)

//     def uscale(self, s):
//         self.transform.isoscale(s)

//     def rotatex(self, d):
//         self.transform.rotatex(d)

//     def rotatey(self, d):
//         self.transform.rotatey(d)

//     def rotatez(self, d):
//         self.transform.rotatez(d)

//     def get_surface(self, i):
//         def yellow(face, u, v):
//             return (0.1, 1.0, 1.0), 0.4, 0.05, 4
//         return yellow

// def atan2(a, b):
//     c = 0.5 * math.atan2(a, b) / math.pi
//     while c < 0.0:
//         c += 1.0
//     return c

impl Node for Sphere {
    fn name(&self) -> &str {
        return "sphere";
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
            ts.push(Intersection::new(scale, t1, transformed_raypos, normalized_transformed_raydir, tr.clone(), IntersectionType::Entry, 0));
        }
        if t2 > 0.0 {
            ts.push(Intersection::new(scale, t2, transformed_raypos, normalized_transformed_raydir, tr.clone(), IntersectionType::Exit, 0));
        }
        ts
    }
    fn inside(&self, pos: Vec3) -> bool {
        let transformed_pos = self.transform.inv_transform_point(pos);
        dot(transformed_pos, transformed_pos) <= 1.0
    }
    fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
        self.transform.translate(tx, ty, tz);
    }
    fn scale(&mut self, _sx: f64, _sy: f64, _sz: f64) {}
    fn uscale(&mut self, _s: f64) {}
    fn rotatex(&mut self, _d: f64) {}
    fn rotatey(&mut self, _d: f64) {}
    fn rotatez(&mut self, _d: f64) {}
}
// class Sphere(Primitive):
//     def intersect(self, raypos, raydir):
//         tr = self.transform
//         raydir = tr.inv_transform_vector(raydir)
//         scale = 1.0 / length(raydir)
//         raydir = mul(raydir, scale) # normalize
//         raypos = tr.inv_transform_point(raypos)
//         s = dot(neg(raypos), raydir)
//         lsq = dot(raypos, raypos)
//         if s < 0.0 and lsq > 1.0:
//             return []
//         msq = lsq - s * s
//         if msq > 1.0:
//             return []
//         q = math.sqrt(1.0 - msq)
//         t1 = s + q
//         t2 = s - q
//         if t1 > t2:
//             t1, t2 = t2, t1
//         ts = []
//         if t1 > 0.0:
//             ts.append(Intersection(scale, t1, raypos, raydir, self, Intersection.ENTRY, 0))
//         if t2 > 0.0:
//             ts.append(Intersection(scale, t2, raypos, raydir, self, Intersection.EXIT, 0))
//         return ts

//     def inside(self, pos):
//         x, y, z = self.transform.inv_transform_point(pos)
//         return (x * x + y * y + z * z) <= 1.0

//     def get_surface(self, i):
//         x, y, z = i.opos
//         v = (y + 1.0) / 2.0
//         u = atan2(x, z)
//         return self.surface(i.face, u, v)

//     def get_normal(self, i):
//         return normalize(self.transform.transform_normal(i.opos))

// class Cube(Primitive):
//     normals = [(0.0, 0.0, -1.0),
//                (0.0, 0.0, 1.0),
//                (-1.0, 0.0, 0.0),
//                (1.0, 0.0, 0.0),
//                (0.0, 1.0, 0.0),
//                (0.0, -1.0, 0.0)]
//     slabs = [(3, 2),
//              (4, 5),
//              (1, 0)]

//     def intersect(self, raypos, raydir):
//         tr = self.transform
//         raydir = tr.inv_transform_vector(raydir)
//         scale = 1.0 / length(raydir)
//         raydir = mul(raydir, scale) # normalize
//         raypos = tr.inv_transform_point(raypos)
//         eps = 1e-15
//         tmin = None
//         tmax = None
//         p = sub((0.5, 0.5, 0.5), raypos)
//         for i in range(3):
//             face1, face2 = self.slabs[i]
//             e = p[i]
//             f = raydir[i]
//             if abs(f) > eps:
//                 t1 = (e + 0.5) / f
//                 t2 = (e - 0.5) / f
//                 if t1 > t2:
//                     t1, t2 = t2, t1
//                     face1, face2 = face2, face1
//                 if tmin is None or t1 > tmin[0]:
//                     tmin = (t1, face1)
//                 if tmax is None or t2 < tmax[0]:
//                     tmax = (t2, face2)
//                 if tmin[0] > tmax[0]:
//                     return []
//                 if tmax[0] < 0.0:
//                     return []
//             elif -e - 0.5 > 0.0 or -e + 0.5 < 0.0:
//                 return []
//         ts = []
//         if tmin[0] > 0.0:
//             ts.append(Intersection(scale, tmin[0], raypos, raydir, self, Intersection.ENTRY, tmin[1]))
//         if tmax[0] > 0.0:
//             ts.append(Intersection(scale, tmax[0], raypos, raydir, self, Intersection.EXIT, tmax[1]))
//         return ts

//     def inside(self, pos):
//         x, y, z = self.transform.inv_transform_point(pos)
//         return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= z <= 1.0

//     def get_surface(self, i):
//         x, y, z = i.opos
//         face = i.face
//         if face == 0:
//             return self.surface(0, x, y)
//         elif face == 1:
//             return self.surface(1, x, y)
//         elif face == 2:
//             return self.surface(2, z, y)
//         elif face == 3:
//             return self.surface(3, z, y)
//         elif face == 4:
//             return self.surface(4, x, z)
//         elif face == 5:
//             return self.surface(5, x, z)
//         else:
//             print opos
//             assert False

//     def get_normal(self, i):
//         return normalize(self.transform.transform_normal(self.normals[i.face]))

// class Cylinder(Primitive):
//     def _solveCyl(self, px, pz, dx, dz):
//         # solve x ^ 2 + z ^ 2 = 1
//         # (px + t * dx) ^ 2 + (pz + t * dz) ^ 2 = 1
//         # a * t ^ 2 + b * t + c = 0
//         # t = (-b +/- sqrt(b ^ 2 - 4 * a * c)) / 2 * a
//         a = dx * dx + dz * dz
//         b = 2 * (px * dx + pz * dz)
//         c = px * px + pz * pz - 1.0
//         sq = b * b - 4 * a * c
//         if sq < 0.0:
//             return None
//         else:
//             root = math.sqrt(sq)
//             t1 = ((-b - root) / (2.0 * a), 0)
//             t2 = ((-b + root) / (2.0 * a), 0)
//             if t1 > t2:
//                 t1, t2 = t2, t1
//             return t1, t2

//     def _solvePlane(self, py, dy):
//         dinv = 1.0 / dy
//         t1 = -py * dinv
//         t2 = (-py + 1.0) * dinv
//         face1 = 2 # bottom
//         face2 = 1 # top
//         tt1 = (t1, face1)
//         tt2 = (t2, face2)
//         if t1 > t2:
//             tt1, tt2 = tt2, tt1
//         return tt1, tt2

//     def intersect(self, raypos, raydir):
//         tr = self.transform
//         raydir = tr.inv_transform_vector(raydir)
//         scale = 1.0 / length(raydir)
//         raydir = mul(raydir, scale) # normalize
//         raypos = tr.inv_transform_point(raypos)
//         eps = 1e-7

//         px, py, pz = raypos
//         dx, dy, dz = raydir
//         ts = []
//         if abs(dy) + eps >= 1.0:
//             # ray is parallel to the cylinder axis
//             frsd = 1.0 - px * px - pz * pz
//             if frsd < 0.0:
//                 # outside cylinder
//                 return []
//             ts = list(self._solvePlane(py, dy))
//         elif abs(dy) < eps:
//             # ray is orthogonal to the cylinder axis
//             # check planes
//             if py < 0.0 or py > 1.0:
//                 return []
//             # check cylinder
//             res = self._solveCyl(px, pz, dx, dz)
//             if not res:
//                 return []
//             ts = list(res)
//         else:
//             # general case
//             # check cylinder
//             res = self._solveCyl(px, pz, dx, dz)
//             if not res:
//                 #print "baz"
//                 return []
//             tc1, tc2  = res
//             # check planes
//             tp1, tp2 = self._solvePlane(py, dy)
//             # same min-max strategy as for cubes
//             # Check max of mins
//             tmin = tp1
//             tmax = tp2
//             if tc1 > tmin:
//                 tmin = tc1
//             if tc2 < tmax:
//                 tmax = tc2
//             if tmin[0] > tmax[0]:
//                 return []
//             if tmax[0] < 0.0:
//                 return []
//             ts = [tmin, tmax]

//         tmin, tmax = ts
//         ts = []
//         if tmin[0] > 0.0:
//             ts.append(Intersection(scale, tmin[0], raypos, raydir, self, Intersection.ENTRY, tmin[1]))
//         if tmax[0] > 0.0:
//             ts.append(Intersection(scale, tmax[0], raypos, raydir, self, Intersection.EXIT, tmax[1]))
//         return ts

//     def inside(self, pos):
//         x, y, z = self.transform.inv_transform_point(pos)
//         return (x * x + z * z) <= 1.0 and 0.0 <= y <= 1.0

//     def get_surface(self, i):
//         x, y, z = i.opos
//         face = i.face
//         if face == 0:
//             return self.surface(0, atan2(x, z), y)
//         elif face == 1:
//             return self.surface(1, (x + 1.0) / 2.0, (z + 1.0) / 2.0)
//         elif face == 2:
//             return self.surface(2, (x + 1.0) / 2.0, (z + 1.0) / 2.0)
//         else:
//             print face
//             raise

//     def get_normal(self, i):
//         if i.face == 0:
//             n = (i.opos[0], 0.0, i.opos[2])
//         elif i.face == 1:
//             n = (0.0, 1.0, 0.0)
//         elif i.face == 2:
//             n =(0.0, -1.0, 0.0)
//         return normalize(self.transform.transform_normal(n))


// class Cone(Primitive):
//     def _solveCone(self, px, py, pz, dx, dy, dz):
//         # solve x ^ 2 + z ^ 2 = y ^ 2
//         # (px + t * dx) ^ 2 + (pz + t * dz) ^ 2 = (py + t * dy) ^ 2
//         # a * t ^ 2 + b * t + c = 0
//         # t = (-b +/- sqrt(b ^ 2 - 4 * a * c)) / 2 * a
//         a = dx * dx + dz * dz - dy * dy
//         b = 2 * (px * dx + pz * dz - py * dy)
//         c = px * px + pz * pz - py * py
//         sq = b * b - 4 * a * c
//         if sq < 0.0:
//             return None
//         else:
//             root = math.sqrt(sq)
//             t1 = ((-b - root) / (2.0 * a), 0)
//             t2 = ((-b + root) / (2.0 * a), 0)
//             if t1 > t2:
//                 t1, t2 = t2, t1
//             return t1, t2

//     def intersect(self, raypos, raydir):
//         tr = self.transform
//         raydir = tr.inv_transform_vector(raydir)
//         scale = 1.0 / length(raydir)
//         raydir = mul(raydir, scale) # normalize
//         raypos = tr.inv_transform_point(raypos)
//         eps = 1e-7

//         px, py, pz = raypos
//         dx, dy, dz = raydir
//         tsc = self._solveCone(px, py, pz, dx, dy, dz)
//         if not tsc:
//             return []
//         tcmin, tcmax = tsc
//         ts = []
//         cminy = py + tcmin[0] * dy
//         cmaxy = py + tcmax[0] * dy
//         if 0.0 <= cminy <= 1.0:
//             ts.append(tcmin)
//         if 0.0 <= cmaxy <= 1.0:
//             ts.append(tcmax)
//         if len(ts) == 0:
//             return []
//         if len(ts) == 2:
//             tr = []
//             if ts[0][0] > 0.0:
//                 tr.append(Intersection(scale, ts[0][0], raypos, raydir, self, Intersection.ENTRY, ts[0][1]))
//             if ts[1][0] > 0.0:
//                 tr.append(Intersection(scale, ts[1][0], raypos, raydir, self, Intersection.EXIT, ts[1][1]))
//             return tr
//         # check plane
//         # since we know there is only one intersection with the cone,
//         # there must be an intersection in the base
//         tp = (-py + 1.0) / dy
//         ts.append((tp, 1))
//         ts.sort()
//         tr = []
//         if ts[0][0] > 0.0:
//             tr.append(Intersection(scale, ts[0][0], raypos, raydir, self, Intersection.ENTRY, ts[0][1]))
//         if ts[1][0] > 0.0:
//             tr.append(Intersection(scale, ts[1][0], raypos, raydir, self, Intersection.EXIT, ts[1][1]))
//         return tr

//     def inside(self, pos):
//         x, y, z = self.transform.inv_transform_point(pos)
//         return (x * x - y * y + z * z) <= 0.0 and 0.0 <= y <= 1.0

//     def get_surface(self, i):
//         x, y, z = i.opos
//         face = i.face
//         if face == 0:
//             return self.surface(0, atan2(x, z), y)
//         elif face == 1:
//             return self.surface(1, (x + 1.0) / 2.0, (z + 1.0) / 2.0)

//     def get_normal(self, i):
//         if i.face == 0:
//             x, y, z = i.opos
//             n = (x, -y, z)
//         elif i.face == 1:
//             n = (0.0, 1.0, 0.0)
//         return normalize(self.transform.transform_normal(n))


// class Plane(Primitive):
//     np = (0.0, 1.0, 0.0)
//     def intersect(self, raypos, raydir):
//         tr = self.transform
//         raydir = tr.inv_transform_vector(raydir)
//         scale = 1.0 / length(raydir)
//         raydir = mul(raydir, scale) # normalize
//         raypos = tr.inv_transform_point(raypos)
//         np = self.np
//         denom = dot(np, raydir)
//         if abs(denom) < 1e-7:
//             return []
//         t = -dot(np, raypos) / denom
//         if t < 0.0:
//             return []
//         if denom > 0.0:
//             return [Intersection(scale, t, raypos, raydir, self, Intersection.EXIT, 0)]
//         else:
//             return [Intersection(scale, t, raypos, raydir, self, Intersection.ENTRY, 0)]

//     def inside(self, pos):
//         return self.transform.inv_transform_py(pos) <= 0.0

//     def get_surface(self, i):
//         x, y, z = i.opos
//         return self.surface(0, x, z)

//     def get_normal(self, i):
//         return normalize(self.transform.transform_normal(self.np))

#[test]
fn test_intersection() {
    let mut i = Intersection::new(
        2.0,
        3.0,
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        Default::default(),
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
        let wpos = i.get_wpos();
        println!("Intersection type: {:?}, pos: {:?}, distance: {:?}", i.t, wpos, i.distance);
    }
}
