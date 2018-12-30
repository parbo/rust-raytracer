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
    pub use raytracer::RendererFactory;

    #[cfg(target_arch = "wasm32")]
    pub fn set_renderer_factory(renderer_factory: Box<raytracer::RendererFactory>) {
        let mut r = raytracer::RENDERER_FACTORY.lock().unwrap();
        *r = renderer_factory;
    }

    pub fn render_gml(gml: &str) {
        evaluator::run(gml).expect("Could not evaluate gml");
    }
}
