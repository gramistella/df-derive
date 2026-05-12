use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;

mod custom {
    pub struct Option<T>(pub T);
}

struct NotAsRef;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_str)]
    label: custom::Option<NotAsRef>,
}

fn main() {}
