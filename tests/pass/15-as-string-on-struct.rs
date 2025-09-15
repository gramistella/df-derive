use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

#[derive(Clone, Debug)]
struct Location {
    city: String,
    country: String,
}

// Implement Display to be used with as_string
impl std::fmt::Display for Location {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}, {}", self.city, self.country)
    }
}

// This struct will derive ToDataFrame, but Display is what matters for as_string.
// Deriving ToDataFrame allows it to be used as a regular nested struct elsewhere.
#[derive(ToDataFrame)]
struct LocationWithCoords {
    lat: f64,
    lon: f64,
}

impl std::fmt::Display for LocationWithCoords {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.lat, self.lon)
    }
}

#[derive(ToDataFrame)]
struct User {
    id: u32,
    #[df_derive(as_string)]
    location: Location, // Does not derive ToDataFrame, must be stringified
    #[df_derive(as_string)]
    coords: LocationWithCoords, // Derives ToDataFrame, but as_string should override flattening
    #[df_derive(as_string)]
    destinations: Vec<Location>,
}

fn main() {
    println!("--- Testing #[df_derive(as_string)] on struct fields ---");

    let user = User {
        id: 1,
        location: Location { city: "New York".to_string(), country: "USA".to_string() },
        coords: LocationWithCoords { lat: 40.7128, lon: -74.0060 },
        destinations: vec![
            Location { city: "London".to_string(), country: "UK".to_string() },
            Location { city: "Tokyo".to_string(), country: "Japan".to_string() },
        ],
    };

    let df = user.to_dataframe().unwrap();
    println!("📊 Resulting DataFrame:\n{}", df);
    
    assert_eq!(df.shape(), (1, 4));
    let expected_names = ["id", "location", "coords", "destinations"];
    assert_eq!(df.get_column_names(), expected_names);

    // Check dtypes
    let schema = df.schema();
    assert_eq!(schema.get("id").unwrap(), &DataType::UInt32);
    assert_eq!(schema.get("location").unwrap(), &DataType::String);
    assert_eq!(schema.get("coords").unwrap(), &DataType::String);
    assert_eq!(schema.get("destinations").unwrap(), &DataType::List(Box::new(DataType::String)));

    // Check values
    assert_eq!(df.column("id").unwrap().get(0).unwrap(), AnyValue::UInt32(1));
    assert_eq!(df.column("location").unwrap().get(0).unwrap(), AnyValue::String("New York, USA"));
    assert_eq!(df.column("coords").unwrap().get(0).unwrap(), AnyValue::String("(40.7128, -74.006)"));
    
    let dests = df.column("destinations").unwrap().get(0).unwrap();
    if let AnyValue::List(series) = dests {
        let ca: Vec<Option<&str>> = series.str().unwrap().into_iter().collect();
        assert_eq!(ca, vec![Some("London, UK"), Some("Tokyo, Japan")]);
    } else {
        panic!("Expected a list for destinations");
    }

    println!("✅ as_string on struct fields test passed!");
}