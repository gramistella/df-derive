// Uses the default `df-derive` facade runtime.

use df_derive::ToDataFrame;
use df_derive::dataframe;

#[derive(Clone, Debug, PartialEq)]
enum Status {
    Active,
    Inactive,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "Active"),
            Self::Inactive => write!(f, "Inactive"),
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
}

fn main() -> polars::prelude::PolarsResult<()> {
    let data = WithEnums {
        status: Status::Active,
        opt_status: Some(Status::Inactive),
        statuses: vec![Status::Active, Status::Inactive, Status::Active],
    };

    let df = <WithEnums as dataframe::ToDataFrame>::to_dataframe(&data)?;
    println!("As string attribute DataFrame:");
    println!("{df}");

    // Show schema to demonstrate string data types
    let schema = <WithEnums as dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (columns use DataType::String or List<String>):");
    for (name, dtype) in schema {
        println!("  {name}: {dtype:?}");
    }

    Ok(())
}
