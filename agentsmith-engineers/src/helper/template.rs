use minijinja::{path_loader, Environment};

pub fn build_template_environment(path: &String) -> Environment {
    let mut env = Environment::new();
    env.set_loader(path_loader(path));
    env
}
