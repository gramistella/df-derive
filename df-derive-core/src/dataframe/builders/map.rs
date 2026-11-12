use std::collections::{BTreeMap, HashMap};

use super::list::{ListOffsets, wrap_columns, wrap_schema};
use crate::dataframe::frame::{child, in_field};
use crate::dataframe::{ColumnBuilder, Columnar, Fields, Record};
use polars::prelude::{Column, PlSmallStr, PolarsResult};
use polars_arrow::bitmap::Bitmap;

/// Encodes a map as two list columns, `<field>.key` and `<field>.value`,
/// with one entry per pair.
///
/// A `BTreeMap` lists its pairs in key order. A `HashMap` lists them in its
/// own iteration order, which is not stable between runs. An empty slot is an
/// empty map.
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

    fn empty<KT, VT>() -> Self
    where
        K: ColumnBuilder<KT>,
        V: ColumnBuilder<VT>,
    {
        Self {
            keys: K::new(),
            values: V::new(),
            offsets: ListOffsets::new(),
        }
    }

    fn reserve_pairs<KT, VT>(&mut self, additional: usize)
    where
        K: ColumnBuilder<KT>,
        V: ColumnBuilder<VT>,
    {
        let pairs = self.offsets.reserve(additional);
        self.keys.reserve(pairs);
        self.values.reserve(pairs);
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
            if let Err(error) = self.keys.push(Some(key)) {
                return Err(in_field(error, "key"));
            }
            if let Err(error) = self.values.push(Some(value)) {
                return Err(in_field(error, "value"));
            }
        }
        self.offsets.push(len)
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

            fn new() -> Self {
                Self::empty::<KT, VT>()
            }

            #[inline(always)]
            fn measure<'a, I>(&mut self, values: I)
            where
                $map<KT, VT $(, $hasher)?>: 'a,
                I: Iterator<Item = Option<&'a $map<KT, VT $(, $hasher)?>>> + Clone,
            {
                let maps = values.flatten();
                self.offsets.measure(maps.clone().map($map::len));
                self.keys.measure(maps.clone().flat_map($map::keys).map(Some));
                self.values.measure(maps.flat_map($map::values).map(Some));
            }

            fn reserve(&mut self, additional: usize) {
                self.reserve_pairs::<KT, VT>(additional);
            }

            #[inline(always)]
            fn push(&mut self, value: Option<&$map<KT, VT $(, $hasher)?>>) -> PolarsResult<()> {
                match value {
                    Some(map) => self.push_pairs(map.iter()),
                    None => self.offsets.push(0),
                }
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
