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

    pub fn render_gml(gml: &str) {
        evaluator::run(gml);
    }
}
