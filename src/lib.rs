#[macro_use]
extern crate lazy_static;

mod evaluator;
mod raytracer;
mod parser;
mod tokenizer;
mod primitives;
mod lights;
mod vecmath;
mod transform;

pub mod render {
    use raytracer;
    use evaluator;
    use std::rc::Rc;

    pub use raytracer::Pixel;
    pub use raytracer::Renderer;

    #[cfg(target_arch = "wasm32")]
    pub fn set_renderer(renderer: Box<raytracer::Renderer>) {
        let mut r = raytracer::RENDERER.lock().unwrap();
        *r = renderer;
    }

    pub fn render_gml(gml: &str) {
        evaluator::run(gml);
    }
}
