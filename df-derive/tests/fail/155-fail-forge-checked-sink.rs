use std::sync::Arc;

use df_derive::dataframe::ColumnSink;
use polars::prelude::Schema;

fn main() {
    let schema = Arc::new(Schema::default());
    let _forged = ColumnSink::new(schema, "forged");
}
