use std::collections::{BTreeMap, HashMap};

use super::list::{ListOffsets, wrap_columns, wrap_schema};
use crate::dataframe::frame::child;
use crate::dataframe::{ColumnBuilder, Columnar, Fields, Record};
use polars::prelude::{Column, PlSmallStr, PolarsResult};
use polars_arrow::bitmap::Bitmap;

/// Encodes a map as two list columns, `<field>.key` and `<field>.value`,
/// with one entry per pair.
///
/// A `BTreeMap` lists its pairs in key order. A `HashMap` lists them in its
/// own iteration order, which is not stable between runs.
pub struct MapBuilder<K, V> {
    keys: K,
    values: V,
    offsets: ListOffsets,
}

impl<K, V> MapBuilder<K, V> {
    fn map_schema<KT, VT>(name: &PlSmallStr, out: &mut Fields)
    where
        K: ColumnBuilder<KT>,
        V: ColumnBuilder<VT>,
    {
        let start = out.len();
        K::schema(child(name, "key"), out);
        V::schema(child(name, "value"), out);
        wrap_schema(out, start);
    }

    fn with_rows<KT, VT>(rows: usize) -> Self
    where
        K: ColumnBuilder<KT>,
        V: ColumnBuilder<VT>,
    {
        Self {
            keys: K::with_capacity(rows),
            values: V::with_capacity(rows),
            offsets: ListOffsets::with_capacity(rows),
        }
    }

    #[inline(always)]
    fn push_pairs<'a, KT: 'a, VT: 'a>(
        &mut self,
        pairs: impl ExactSizeIterator<Item = (&'a KT, &'a VT)>,
    ) -> PolarsResult<()>
    where
        K: ColumnBuilder<KT>,
        V: ColumnBuilder<VT>,
    {
        let len = pairs.len();
        for (key, value) in pairs {
            self.keys.push(key)?;
            self.values.push(value)?;
        }
        self.offsets.push(len);
        Ok(())
    }

    fn finish_map<KT, VT>(
        self,
        name: &PlSmallStr,
        validity: Option<&Bitmap>,
        out: &mut Vec<Column>,
    ) -> PolarsResult<()>
    where
        K: ColumnBuilder<KT>,
        V: ColumnBuilder<VT>,
    {
        let start = out.len();
        self.keys.finish(child(name, "key"), None, out)?;
        self.values.finish(child(name, "value"), None, out)?;
        wrap_columns(out, start, self.offsets, validity)
    }
}

macro_rules! map {
    ($map:ident<KT, VT $(, $hasher:ident)?>) => {
        impl<KT, VT, K, V $(, $hasher)?> ColumnBuilder<$map<KT, VT $(, $hasher)?>>
            for MapBuilder<K, V>
        where
            K: ColumnBuilder<KT>,
            V: ColumnBuilder<VT>,
        {
            fn schema(name: PlSmallStr, out: &mut Fields) {
                Self::map_schema::<KT, VT>(&name, out);
            }

            const LISTS: usize = 1 + K::LISTS + V::LISTS;

            fn with_capacity(rows: usize) -> Self {
                Self::with_rows::<KT, VT>(rows)
            }

            #[inline(always)]
            fn measure(value: &$map<KT, VT $(, $hasher)?>, counts: &mut [usize]) {
                counts[0] += value.len();
                if K::LISTS + V::LISTS > 0 {
                    let (keys, values) = counts[1..].split_at_mut(K::LISTS);
                    for (key, value) in value {
                        K::measure(key, keys);
                        V::measure(value, values);
                    }
                }
            }

            fn with_capacities(rows: usize, counts: &[usize]) -> Self {
                let (keys, values) = counts[1..].split_at(K::LISTS);
                Self {
                    keys: K::with_capacities(counts[0], keys),
                    values: V::with_capacities(counts[0], values),
                    offsets: ListOffsets::with_capacity(rows),
                }
            }

            #[inline(always)]
            fn push(&mut self, value: &$map<KT, VT $(, $hasher)?>) -> PolarsResult<()> {
                self.push_pairs(value.iter())
            }

            #[inline(always)]
            fn push_null(&mut self) {
                self.offsets.push_empty();
            }

            fn finish(
                self,
                name: PlSmallStr,
                validity: Option<&Bitmap>,
                out: &mut Vec<Column>,
            ) -> PolarsResult<()> {
                self.finish_map::<KT, VT>(&name, validity, out)
            }
        }

        impl<KT: Columnar, VT: Columnar $(, $hasher)?> Columnar for $map<KT, VT $(, $hasher)?> {
            type Builder = MapBuilder<KT::Builder, VT::Builder>;
        }

        impl<KT: Columnar, VT: Columnar $(, $hasher)?> Record for $map<KT, VT $(, $hasher)?> {}
    };
}

map!(HashMap<KT, VT, S>);
map!(BTreeMap<KT, VT>);
