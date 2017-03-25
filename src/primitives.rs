use vecmath::{Vec3};
use transform::{Transform};

pub struct Intersection {
}

pub trait Node {
    fn intersect(&self, raypos: Vec3, raydir: Vec3) -> Vec<Intersection>;
    fn inside(&self, pos: Vec3) -> bool;
    fn translate(&mut self, tx: f64, ty: f64, tz: f64);
    fn scale(&mut self, sx: f64, sy: f64, sz: f64);
    fn uscale(&mut self, s: f64);
    fn rotatex(&mut self, d: f64);
    fn rotatey(&mut self, d: f64);
    fn rotatez(&mut self, d: f64);
}

pub struct Operator {
    obj1: Box<Node>,
    obj2: Box<Node>,
    rule: Fn(bool, bool) -> bool
}

pub struct Sphere {
    transform: Transform
}

// class Intersection(object):
//     ENTRY = 0
//     EXIT = 1
//     def __init__(self, scale, odistance, rp, rd, primitive, t, face):
//         self.scale = scale
//         self.odistance = odistance
//         self.distance = scale * odistance
//         self.rp = rp
//         self.rd = rd
//         self.primitive = primitive
//         self.t = t
//         self.face = face
//         self._wpos = None
//         self._opos = None
//         self._normal = None

//     def __cmp__(self, rhs):
//         return cmp(self.distance, rhs.distance)

//     def __str__(self):
//         return "%s %s %d"%(self.distance, self.primitive, self.t)
//         return "%s %s %s %s %s %s"%(self.distance, self.wpos, self.opos, self.normal, self.primitive, self.t)

//     def switch(self, t):
//         if self.t != t:
//             self.t = t
//             self._normal = neg(self.normal)

//     def opos():
//         def fget(self):
//             if not self._opos:
//                 self._opos = add(self.rp, mul(self.rd, self.odistance))
//             return self._opos
//         return locals()
//     opos = property(**opos())

//     def wpos():
//         def fget(self):
//             opos = self.opos
//             if not self._wpos:
//                 self._wpos = self.primitive.transform.transform_point(opos)
//             return self._wpos
//         return locals()
//     wpos = property(**wpos())

//     def normal():
//         def fget(self):
//             if not self._normal:
//                 self._normal = self.primitive.get_normal(self)
//             return self._normal
//         return locals()
//     normal = property(**normal())


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
    fn intersect(&self, raypos: Vec3, raydir: Vec3) -> Vec<Intersection> {
        vec!()
    }
    fn inside(&self, pos: Vec3) -> bool {
        false
    }
    fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
    }
    fn scale(&mut self, sx: f64, sy: f64, sz: f64) {
    }
    fn uscale(&mut self, s: f64) {
    }
    fn rotatex(&mut self, d: f64) {
    }
    fn rotatey(&mut self, d: f64) {
    }
    fn rotatez(&mut self, d: f64) {
    }
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
