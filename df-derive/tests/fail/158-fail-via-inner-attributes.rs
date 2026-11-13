use df_derive::ToDataFrame;

// A type with `via` encodes as its row type. Its own fields and variants are
// never read, so an attribute on one of them would do nothing.

#[derive(ToDataFrame)]
struct PointRow {
    x: f64,
}

impl From<&Point> for PointRow {
    fn from(point: &Point) -> Self {
        Self { x: point.x }
    }
}

impl From<&Shape> for PointRow {
    fn from(_: &Shape) -> Self {
        Self { x: 0.0 }
    }
}

#[derive(ToDataFrame)]
#[df_derive(via = "PointRow")]
struct Point {
    #[df_derive(skip)]
    x: f64,
    label: String,
}

#[derive(ToDataFrame)]
#[df_derive(via = "PointRow")]
enum Shape {
    Circle,
    Polygon {
        #[df_derive(as_string)]
        sides: u32,
    },
}

fn main() {}
