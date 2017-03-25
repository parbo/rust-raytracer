use vecmath::{normalize, add, sub, neg, mul, dot, length, cross, Vec3};
use std::path::Path;
use std::io::{Write, Result};
use std::fs::File;

type Pixel = [f64; 3];

fn write_ppm_file(pixels: &[Pixel], w: u32, h: u32, filename: &str) -> Result<()> {
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

fn trace(// amb, lights, obj,
         depth: u32,
         raypos: Vec3,
         raydir: Vec3)
         -> Pixel {
    [1.0, 0.0, 0.0]
}

fn render(// amb, lights, obj,
          depth: u32,
          fov: f64,
          w: u32,
          h: u32,
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
            let p = trace(// amb, lights, obj,
                          depth,
                          raypos,
                          raydir);
            pixels.push(p);
        }
    }
    write_ppm_file(&pixels, w, h, filename);
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
    write_ppm_file(&pixels, 256, 256, "test.ppm");
}

#[test]
fn test_raytrace() {
    render(3, 90.0, 256, 256, "raytrace.ppm");
}
