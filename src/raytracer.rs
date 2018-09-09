use vecmath::{normalize, Vec3, mul, cmul};
use std::path::Path;
use std::io::{Write, Result};
use std::fs::File;
use std::rc::Rc;
use primitives::{Node, Sphere, IntersectionType};
use lights::{Light, DirectionalLight};

type Pixel = [f64; 3];
type Color = [f64; 3];

fn write_ppm_file(pixels: &[Pixel], w: i64, h: i64, filename: &str) -> Result<()> {
    let path = Path::new(filename);
    let mut file = try!(File::create(&path));
    let header = format!("P6 {} {} 255\n", w, h);
    try!(file.write(header.as_bytes()));
    for p in pixels {
        let p_bytes = [(255.0 * p[0].max(0.0).min(1.0)) as u8,
                       (255.0 * p[1].max(0.0).min(1.0)) as u8,
                       (255.0 * p[2].max(0.0).min(1.0)) as u8];
        try!(file.write(&p_bytes));
    }
    Ok(())
}

fn get_ambient(c: Vec3, ia: Vec3, kd: f64) -> Color {
    return mul(cmul(ia, c), kd)
}

fn trace(amb: Vec3,
         _lights: &[Box<Light>],
         scene: &Node,
         _depth: i64,
         raypos: Vec3,
         raydir: Vec3)
         -> Pixel {
    let mut i = scene.intersect(raypos, raydir);
    // for ii in i.iter_mut() {
    //     let wpos = ii.get_wpos();
    //     println!("Intersection type: {:?}, pos: {:?}, distance: {:?}", ii.t, wpos, ii.distance);
    // }
    if i.len() > 0 {
        let ref isect = &i[0];
        if isect.t == IntersectionType::Exit {
            return [0.0, 0.0, 0.0];
        }
//        let (sc, kd, ks, n) = isect.primitive.get_surface(isect);
        let (sc, kd, _ks, _n) =  ([0.1, 0.1, 1.0], 0.3, 0.2, 6.0);
        let c = get_ambient(sc, amb, kd);
        return c;  // No lights
        // diffuse = (0.0, 0.0, 0.0)
        // specular = (0.0, 0.0, 0.0)
        // pos = isect.wpos
        // normal = isect.normal
        // for light in lights:
        //     lightdir, lightdistance = light.get_direction(pos)
        //     df = dot(normal, lightdir)
        //     if df > 0.0:
        //         poseps = add(pos, mul(lightdir, 1e-7))
        //         i = scene.intersect(poseps, lightdir)
        //         if not i or (lightdistance and (lightdistance < i[0].distance)):
        //             ic = cmul(sc, light.get_intensity(pos))
        //             if kd > 0.0:
        //                 diffuse = add(diffuse, mul(ic, df))
        //             if ks > 0.0:
        //                 specular = add(specular, get_specular(ic, lightdir, normal, pos, raypos, n))
        // c = add(c, add(mul(diffuse, kd), mul(specular, ks)))
        // if ks > 0.0 and depth > 0:
        //     refl_raydir = normalize(sub(raydir, mul(normal, 2 * dot(raydir, normal))))
        //     poseps = add(pos, mul(refl_raydir, 1e-7))
        //     rc = trace(amb, lights, scene, depth - 1, poseps, refl_raydir)
        //     return add(c, mul(cmul(rc, sc), ks))
        // else:
        //     return c
    } else {
        return [0.0, 0.0, 0.0];
    }
}

pub fn render(amb: Vec3,
              lights: Vec<Box<Light>>,
              scene: Box<Node>,
              depth: i64,
              fov: f64,
              w: i64,
              h: i64,
              filename: &str) {
    let mut pixels = Vec::new();
    let raypos = [0.0, 0.0, -1.0];
    let w_world = 2.0 * (0.5 * fov.to_radians()).tan();
    let h_world = h as f64 * w_world / w as f64;
    let c_x = -0.5 * w_world;
    let c_y = 0.5 * h_world;
    let pw = w_world / w as f64;
    for iy in 0..h {
        let y = iy as f64;
        for ix in 0..w {
            let x = ix as f64;
            let dir = [c_x + (x + 0.5) * pw, c_y - (y + 0.5) * pw, -raypos[2]];
            let raydir = normalize(dir);
            let p = trace(amb,
                          &lights,
                          &*scene,
                          depth,
                          raypos,
                          raydir);
            pixels.push(p);
        }
    }
    write_ppm_file(&pixels, w, h, filename).expect("failed to write file");
}

#[test]
fn test_normalize() {
    assert_eq!(normalize([7.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
}

#[test]
fn test_ppm() {
    let mut pixels = Vec::new();
    for y in 0..256 {
        for x in 0..256 {
            pixels.push([x as f64 / 255.0, y as f64 / 255.0, (x + y) as f64 / (2.0 * 255.0)]);
        }
    }
    write_ppm_file(&pixels, 256, 256, "test.ppm").expect("failed to write file");
}

#[cfg(test)]
fn render_scene(lights: Vec<Box<Light>>, mut scene: Box<Node>, name: &str) {
    scene.translate(0.0, 0.0, 3.0);
    render([1.0, 1.0, 1.0], lights, scene, 3, 90.0, 256, 256, name);
}

#[test]
fn test_raytrace() {
    let mut lights : Vec<Box<Light>> = Vec::new();
    lights.push(Box::new(DirectionalLight::new([1.0, 0.0, 0.0], [0.3, 0.4, 0.5])));
    let scene = Box::new(Sphere::new(Rc::new(Box::new(|_face, _u, _v| ([1.0, 0.0, 0.0], 0.9, 0.9, 0.9)))));
    render_scene(lights, scene, "scene_sphere.ppm");
}
