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
    #[cfg(target_arch = "wasm32")]
    pub use raytracer::RendererFactory;

    pub use raytracer::Pixel;
    pub use raytracer::Renderer;

    #[cfg(target_arch = "wasm32")]
    pub fn set_renderer_factory(renderer_factory: Box<dyn raytracer::RendererFactory>) {
        raytracer::RENDERER_FACTORY.with(|r| {
            *r.borrow_mut() = renderer_factory;
        });
    }

    pub fn render_gml(gml: &str) {
        evaluator::run(gml).expect("Could not evaluate gml");
    }
}
