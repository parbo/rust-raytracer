#[cfg(target_arch = "wasm32")]
#[macro_use]
extern crate lazy_static;

mod evaluator;
mod lights;
mod parser;
mod primitives;
mod raytracer;
mod tokenizer;
mod transform;
mod vecmath;

pub mod render {
    use evaluator;

    #[cfg(target_arch = "wasm32")]
    use raytracer;

    pub use raytracer::Pixel;
    pub use raytracer::Renderer;

    #[cfg(target_arch = "wasm32")]
    pub fn set_renderer(renderer: Box<raytracer::Renderer>) {
        let mut r = raytracer::RENDERER.lock().unwrap();
        *r = renderer;
    }

    pub fn render_gml(gml: &str) {
        evaluator::run(gml).expect("Could not evaluate gml");
    }
}
