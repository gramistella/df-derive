use df_derive::ToDataFrame;

mod custom {
    pub struct Option<T>(pub T);
}

struct NotDisplay;

#[derive(ToDataFrame)]
struct Bad {
    #[df_derive(as_string)]
    label: custom::Option<NotDisplay>,
}

fn main() {}
