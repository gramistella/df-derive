use df_derive::ToDataFrame;

#[allow(dead_code)]
mod dataframe {
    use polars::prelude::{DataFrame, DataType, PolarsResult};

    pub trait ToDataFrame {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
        fn empty_dataframe() -> PolarsResult<DataFrame>;
        fn schema() -> PolarsResult<Vec<(&'static str, DataType)>>;
    }

    pub trait Columnar: Sized {
        fn columnar_to_dataframe(items: &[Self]) -> PolarsResult<DataFrame>;
    }

    pub trait ToDataFrameVec {
        fn to_dataframe(&self) -> PolarsResult<DataFrame>;
    }

    impl<T> ToDataFrameVec for [T]
    where
        T: Columnar + ToDataFrame,
    {
        fn to_dataframe(&self) -> PolarsResult<DataFrame> {
            if self.is_empty() {
                return <T as ToDataFrame>::empty_dataframe();
            }
            <T as Columnar>::columnar_to_dataframe(self)
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Status { Active, Inactive }

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Status::Active => write!(f, "Active"),
            Status::Inactive => write!(f, "Inactive"),
        }
    }
}

#[derive(ToDataFrame)]
#[df_derive(trait = "crate::dataframe::ToDataFrame")]
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
    
    let df = <WithEnums as crate::dataframe::ToDataFrame>::to_dataframe(&data)?;
    println!("As string attribute DataFrame:");
    println!("{}", df);
    
    // Show schema to demonstrate string data types
    let schema = <WithEnums as crate::dataframe::ToDataFrame>::schema()?;
    println!("\nSchema (columns use DataType::String or List<String>):");
    for (name, dtype) in schema {
        println!("  {}: {:?}", name, dtype);
    }

    Ok(())
}
