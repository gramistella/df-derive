use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

#[derive(Clone, Debug)]
struct Location {
    code: String,
}

// AsRef<str> instead of Display: this is what `as_str` requires.
impl AsRef<str> for Location {
    fn as_ref(&self) -> &str {
        &self.code
    }
}

// LocationWithCoords derives ToDataFrame, but as_str should override the
// nested-flattening behavior just like as_string does.
#[derive(ToDataFrame, Clone)]
struct LocationWithCoords {
    lat: f64,
    lon: f64,
}

impl AsRef<str> for LocationWithCoords {
    fn as_ref(&self) -> &str {
        "WITH_COORDS"
    }
}

#[derive(ToDataFrame)]
struct User {
    id: u32,
    #[df_derive(as_str)]
    location: Location,
    #[df_derive(as_str)]
    coords: LocationWithCoords,
    #[df_derive(as_str)]
    destinations: Vec<Location>,
    #[df_derive(as_str)]
    opt_destinations: Option<Vec<Location>>,
}

fn main() {
    println!("--- Testing #[df_derive(as_str)] on struct fields ---");

    let user = User {
        id: 1,
        location: Location {
            code: "NYC".to_string(),
        },
        coords: LocationWithCoords {
            lat: 40.7128,
            lon: -74.0060,
        },
        destinations: vec![
            Location {
                code: "LON".to_string(),
            },
            Location {
                code: "TYO".to_string(),
            },
        ],
        opt_destinations: Some(vec![
            Location {
                code: "PAR".to_string(),
            },
            Location {
                code: "BER".to_string(),
            },
        ]),
    };

    let df = user.to_dataframe().unwrap();
    println!("📊 Resulting DataFrame:\n{}", df);

    assert_eq!(df.shape(), (1, 5));
    let expected_names = ["id", "location", "coords", "destinations", "opt_destinations"];
    assert_eq!(df.get_column_names(), expected_names);

    let schema = df.schema();
    assert_eq!(schema.get("id").unwrap(), &DataType::UInt32);
    assert_eq!(schema.get("location").unwrap(), &DataType::String);
    assert_eq!(schema.get("coords").unwrap(), &DataType::String);
    assert_eq!(
        schema.get("destinations").unwrap(),
        &DataType::List(Box::new(DataType::String))
    );
    assert_eq!(
        schema.get("opt_destinations").unwrap(),
        &DataType::List(Box::new(DataType::String))
    );

    assert_eq!(
        df.column("id").unwrap().get(0).unwrap(),
        AnyValue::UInt32(1)
    );
    assert_eq!(
        df.column("location").unwrap().get(0).unwrap(),
        AnyValue::String("NYC")
    );
    assert_eq!(
        df.column("coords").unwrap().get(0).unwrap(),
        AnyValue::String("WITH_COORDS")
    );

    let dests = df.column("destinations").unwrap().get(0).unwrap();
    if let AnyValue::List(series) = dests {
        let ca: Vec<Option<&str>> = series.str().unwrap().into_iter().collect();
        assert_eq!(ca, vec![Some("LON"), Some("TYO")]);
    } else {
        panic!("Expected a list for destinations");
    }

    let opt_dests = df.column("opt_destinations").unwrap().get(0).unwrap();
    if let AnyValue::List(series) = opt_dests {
        let ca: Vec<Option<&str>> = series.str().unwrap().into_iter().collect();
        assert_eq!(ca, vec![Some("PAR"), Some("BER")]);
    } else {
        panic!("Expected a list for opt_destinations");
    }

    println!("✅ as_str on struct fields test passed!");
}
