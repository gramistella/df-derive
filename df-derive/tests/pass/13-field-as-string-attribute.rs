use df_derive::ToDataFrame;
use polars::prelude::*;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;

#[derive(Clone, Debug, PartialEq)]
enum Status {
    Active,
    Inactive,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::Active => write!(f, "ACTIVE"),
            Status::Inactive => write!(f, "INACTIVE"),
        }
    }
}

#[derive(ToDataFrame)]
struct WithEnums {
    #[df_derive(as_string)]
    status: Status,
    #[df_derive(as_string)]
    opt_status: Option<Status>,
    #[df_derive(as_string)]
    statuses: Vec<Status>,
    #[df_derive(as_string)]
    opt_statuses: Option<Vec<Status>>,
}

fn assert_col_str(df: &DataFrame, col: &str, expected: &str) {
    let v = df.column(col).unwrap().get(0).unwrap();
    match v {
        AnyValue::String(s) => assert_eq!(s, expected),
        AnyValue::StringOwned(ref s) => assert_eq!(s, expected),
        _ => panic!("unexpected AnyValue for {}: {:?}", col, v),
    }
}

fn main() {
    println!("--- Testing #[df_derive(as_string)] attribute for enum serialization ---");
    
    // Non-empty case
    println!("🔄 Creating test data with enum values...");
    let s = WithEnums {
        status: Status::Active,
        opt_status: Some(Status::Inactive),
        statuses: vec![Status::Active, Status::Inactive],
        opt_statuses: Some(vec![Status::Inactive, Status::Active]),
    };

    println!("🔄 Converting to DataFrame...");
    let df = s.to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 4));
    
    println!("📊 Resulting DataFrame:");
    println!("{}", df);
    
    println!("🔄 Verifying schema types (all should be String/List<String>)...");
    let schema = df.schema();
    assert_eq!(schema.get("status"), Some(&DataType::String));
    assert_eq!(schema.get("opt_status"), Some(&DataType::String));
    assert_eq!(schema.get("statuses"), Some(&DataType::List(Box::new(DataType::String))));
    assert_eq!(schema.get("opt_statuses"), Some(&DataType::List(Box::new(DataType::String))));

    println!("🔄 Verifying enum values are serialized as strings...");
    assert_col_str(&df, "status", "ACTIVE");
    assert_col_str(&df, "opt_status", "INACTIVE");
    println!("🔄 Verifying Vec<Status> serialization...");
    {
        let av = df.column("statuses").unwrap().get(0).unwrap();
        if let AnyValue::List(inner) = av {
            let vals: Vec<String> = inner
                .iter()
                .map(|v| match v {
                    AnyValue::String(s) => s.to_string(),
                    AnyValue::StringOwned(ref s) => s.to_string(),
                    other => panic!("unexpected AnyValue in statuses: {:?}", other),
                })
                .collect();
            assert_eq!(vals, vec!["ACTIVE", "INACTIVE"]);
        } else {
            panic!("expected List for statuses")
        }
    }
    
    println!("🔄 Verifying Option<Vec<Status>> serialization...");
    {
        let av = df.column("opt_statuses").unwrap().get(0).unwrap();
        if let AnyValue::List(inner) = av {
            let vals: Vec<String> = inner
                .iter()
                .map(|v| match v {
                    AnyValue::String(s) => s.to_string(),
                    AnyValue::StringOwned(ref s) => s.to_string(),
                    other => panic!("unexpected AnyValue in opt_statuses: {:?}", other),
                })
                .collect();
            assert_eq!(vals, vec!["INACTIVE", "ACTIVE"]);
        } else {
            panic!("expected List for opt_statuses")
        }
    }

    // Empty DF schema
    println!("🔄 Testing empty DataFrame schema consistency...");
    let empty = WithEnums::empty_dataframe().unwrap();
    assert_eq!(empty.shape(), (0, 4));
    let schema = empty.schema();
    assert_eq!(schema.get("status"), Some(&DataType::String));
    assert_eq!(schema.get("opt_status"), Some(&DataType::String));
    assert_eq!(schema.get("statuses"), Some(&DataType::List(Box::new(DataType::String))));
    assert_eq!(schema.get("opt_statuses"), Some(&DataType::List(Box::new(DataType::String))));
    
    println!("\n✅ #[df_derive(as_string)] attribute test completed successfully!");
}


