use lights::Light;
use primitives::{IntersectionType, Node};
use vecmath::{add, cmul, dot, mul, neg, normalize, sub, Vec3};

use std::cell::RefCell;

pub type Pixel = [f64; 3];
pub type Color = [f64; 3];

#[cfg(not(target_arch = "wasm32"))]
fn write_ppm_file(pixels: &[Pixel], w: i64, h: i64, filename: &str) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;
    let path = Path::new(filename);
    let mut file = File::create(&path)?;
    let header = format!("P6 {} {} 255\n", w, h);
    let mut data = vec![];
    data.reserve(header.len() + pixels.len() * 3);
    data.extend(header.as_bytes());
    for p in pixels {
        data.push((255.0 * p[0].max(0.0).min(1.0)) as u8);
        data.push((255.0 * p[1].max(0.0).min(1.0)) as u8);
        data.push((255.0 * p[2].max(0.0).min(1.0)) as u8);
    }
    file.write_all(&data)?;
    Ok(())
}

fn get_ambient(c: Vec3, ia: Vec3, kd: f64) -> Color {
    mul(cmul(ia, c), kd)
}

fn get_specular(ic: Color, lightdir: Vec3, sn: Vec3, pos: Vec3, raypos: Vec3, n: f64) -> Color {
    let halfway = normalize(add(lightdir, normalize(sub(raypos, pos))));
    let sp = dot(sn, halfway);
    if sp > 0.0 {
        mul(ic, sp.powf(n))
    } else {
        [0.0, 0.0, 0.0]
    }
}

fn trace(
    amb: Vec3,
    lights: &[Box<dyn Light>],
    scene: &dyn Node,
    depth: i64,
    raypos: Vec3,
    raydir: Vec3,
) -> Pixel {
    let i = scene.intersect(raypos, raydir);
    if !i.is_empty() {
        let isect = &(&i[0]);
        if isect.t == IntersectionType::Exit {
            return [0.0, 0.0, 0.0];
        }
        let primitive = scene.find_primitive(isect.primitive_id).unwrap();
        let opos = isect.get_opos();
        let (sc, kd, ks, n) = primitive.get_surface(opos, isect.face).unwrap();
        let mut c = get_ambient(sc, amb, kd);
        let mut diffuse = [0.0, 0.0, 0.0];
        let mut specular = [0.0, 0.0, 0.0];
        let pos = primitive.transform_point(opos);
        let mut normal = primitive.get_normal(opos, isect.face).unwrap();
        if isect.switched() {
            normal = neg(normal);
        }
        for light in lights.iter() {
            let (lightdir, lightdistance) = light.get_direction(pos);
            let df = dot(normal, lightdir);
            if df > 0.0 {
                let poseps = add(pos, mul(lightdir, 1e-7));
                let lighti = scene.intersect(poseps, lightdir);
                // This must be possible to do more nicely
                let mut do_lights = lighti.is_empty();
                if !do_lights {
                    if let Some(ld) = lightdistance {
                        if ld < lighti[0].distance {
                            do_lights = true;
                        }
                    }
                }
                if do_lights {
                    let ic = cmul(sc, light.get_intensity(pos));
                    if kd > 0.0 {
                        diffuse = add(diffuse, mul(ic, df));
                    }
                    if ks > 0.0 {
                        specular =
                            add(specular, get_specular(ic, lightdir, normal, pos, raypos, n));
                    }
                }
            }
        }
        let diff = mul(diffuse, kd);
        let spec = mul(specular, ks);
        let combined = add(diff, spec);
        c = add(c, combined);
        if ks > 0.0 && depth > 0 {
            let refl_raydir = normalize(sub(raydir, mul(normal, 2.0 * dot(raydir, normal))));
            let poseps = add(pos, mul(refl_raydir, 1e-7));
            let rc = trace(amb, lights, scene, depth - 1, poseps, refl_raydir);
            add(c, mul(cmul(rc, sc), ks))
        } else {
            c
        }
    } else {
        [0.0, 0.0, 0.0]
    }
}

pub trait Renderer {
    fn new_image(&mut self, name: &str, w: i64, h: i64);
    fn push_pixel(&mut self, pixel: Pixel);
    fn done(&mut self);
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_pass_by_value)]
pub fn render_pixels(
    amb: Vec3,
    lights: &[Box<dyn Light>],
    scene: Box<dyn Node>,
    depth: i64,
    fov: f64,
    w: i64,
    h: i64,
    renderer: &mut dyn Renderer,
) {
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
            let pixel = trace(amb, &lights, &*scene, depth, raypos, raydir);
            renderer.push_pixel(pixel);
        }
    }
}

struct VecImage {
    name: String,
    w: i64,
    h: i64,
    pixels: Vec<Pixel>,
}

impl VecImage {
    fn new() -> VecImage {
        VecImage {
            name: "".to_string(),
            w: 0,
            h: 0,
            pixels: Vec::new(),
        }
    }
}

impl Renderer for VecImage {
    fn new_image(&mut self, name: &str, w: i64, h: i64) {
        self.name = name.to_string();
        self.w = w;
        self.h = h;
        self.pixels.resize((w * h) as usize, [0.0, 0.0, 0.0]);
    }
    fn push_pixel(&mut self, p: Pixel) {
        self.pixels.push(p);
    }
    fn done(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        write_ppm_file(&self.pixels, self.w, self.h, &self.name).expect("failed to write file");
    }
}

pub trait RendererFactory {
    fn create(&self) -> Box<dyn Renderer>;
}

struct DefaultRendererFactory {}

impl DefaultRendererFactory {
    fn new() -> DefaultRendererFactory {
        DefaultRendererFactory {}
    }
}

impl RendererFactory for DefaultRendererFactory {
    fn create(&self) -> Box<dyn Renderer> {
        Box::new(VecImage::new())
    }
}

thread_local! {
    pub static RENDERER_FACTORY: RefCell<Box<dyn RendererFactory>> = RefCell::new(Box::new(DefaultRendererFactory::new()))
}

#[allow(clippy::too_many_arguments)]
pub fn render(
    amb: Vec3,
    lights: &[Box<dyn Light>],
    scene: Box<dyn Node>,
    depth: i64,
    fov: f64,
    w: i64,
    h: i64,
    filename: &str,
) {
    RENDERER_FACTORY.with(|renderer_factory| {
        let mut renderer = renderer_factory.borrow().create();
        renderer.new_image(filename, w, h);
        render_pixels(amb, lights, scene, depth, fov, w, h, &mut *renderer);
    });
}

#[cfg(test)]
mod test {
    use super::*;
    use lights::DirectionalLight;
    use primitives::Sphere;
    use std::rc::Rc;

    #[test]
    fn test_normalize() {
        assert_eq!(normalize([7.0, 0.0, 0.0]), [1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ppm() {
        let mut pixels = Vec::new();
        for y in 0..256 {
            for x in 0..256 {
                pixels.push([
                    x as f64 / 255.0,
                    y as f64 / 255.0,
                    (x + y) as f64 / (2.0 * 255.0),
                ]);
            }
        }
        write_ppm_file(&pixels, 256, 256, "test.ppm").expect("failed to write file");
    }

    fn render_scene(lights: &[Box<dyn Light>], mut scene: Box<dyn Node>, name: &str) {
        scene.translate(0.0, 0.0, 3.0);
        render([1.0, 1.0, 1.0], lights, scene, 3, 90.0, 256, 256, name);
    }

    #[test]
    fn test_raytrace() {
        let mut lights: Vec<Box<dyn Light>> = Vec::new();
        lights.push(Box::new(DirectionalLight::new(
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.5],
        )));
        let scene = Box::new(Sphere::new(Rc::new(Box::new(|_face, _u, _v| {
            ([1.0, 0.0, 0.0], 0.9, 0.9, 0.9)
        }))));
        render_scene(&lights, scene, "scene_sphere.ppm");
    }
}
