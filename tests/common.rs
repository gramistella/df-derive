// Test helper that gives older tests their historical
// `crate::core::dataframe::*` import path while exercising the default
// facade runtime trait identity.

pub mod dataframe {
    #[allow(unused_imports)]
    pub use df_derive::dataframe::*;
}
