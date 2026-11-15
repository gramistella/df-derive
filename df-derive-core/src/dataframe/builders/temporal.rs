use std::marker::PhantomData;

use polars::prelude::{
    Column, DataType, Int64Chunked, IntoSeries, PlSmallStr, PolarsError, PolarsResult, TimeUnit,
};
use polars_arrow::bitmap::Bitmap;

use super::primitive::{primitive_array, try_extend};
use crate::dataframe::{ColumnBuilder, Columnar, Fields};

#[cold]
#[inline(never)]
fn out_of_range(message: &'static str) -> PolarsError {
    PolarsError::ComputeError(message.into())
}

/// A Polars time unit chosen at the type level.
///
/// `#[df_derive(time_unit = "ms" | "us" | "ns")]` selects [`Ms`], [`Us`] or
/// [`Ns`].
pub trait Unit: 'static {
    /// The Polars time unit.
    const UNIT: TimeUnit;

    /// Whole units in a `std::time::Duration`.
    ///
    /// # Errors
    /// Returns an error when the count does not fit in `i64`.
    fn from_std(value: &std::time::Duration) -> PolarsResult<i64>;

    /// Whole units since the Unix epoch.
    ///
    /// # Errors
    /// Returns an error when the instant is outside the unit's range.
    #[cfg(feature = "chrono")]
    fn from_instant<Tz: chrono::TimeZone>(value: &chrono::DateTime<Tz>) -> PolarsResult<i64>;

    /// Whole units since the Unix epoch, reading the value as UTC.
    ///
    /// # Errors
    /// Returns an error when the instant is outside the unit's range.
    #[cfg(feature = "chrono")]
    fn from_naive(value: &chrono::NaiveDateTime) -> PolarsResult<i64>;

    /// Whole units in a `chrono::TimeDelta`.
    ///
    /// # Errors
    /// Returns an error when the count does not fit in `i64`.
    #[cfg(feature = "chrono")]
    fn from_delta(value: &chrono::TimeDelta) -> PolarsResult<i64>;
}

/// Milliseconds.
pub struct Ms;
/// Microseconds.
pub struct Us;
/// Nanoseconds.
pub struct Ns;

impl Unit for Ms {
    const UNIT: TimeUnit = TimeUnit::Milliseconds;

    #[inline(always)]
    fn from_std(value: &std::time::Duration) -> PolarsResult<i64> {
        i64::try_from(value.as_millis())
            .map_err(|_| out_of_range("df-derive: std::time::Duration value out of i64 ms range"))
    }

    #[cfg(feature = "chrono")]
    #[inline(always)]
    fn from_instant<Tz: chrono::TimeZone>(value: &chrono::DateTime<Tz>) -> PolarsResult<i64> {
        Ok(value.timestamp_millis())
    }

    #[cfg(feature = "chrono")]
    #[inline(always)]
    fn from_naive(value: &chrono::NaiveDateTime) -> PolarsResult<i64> {
        Ok(value.and_utc().timestamp_millis())
    }

    #[cfg(feature = "chrono")]
    #[inline(always)]
    fn from_delta(value: &chrono::TimeDelta) -> PolarsResult<i64> {
        Ok(value.num_milliseconds())
    }
}

impl Unit for Us {
    const UNIT: TimeUnit = TimeUnit::Microseconds;

    #[inline(always)]
    fn from_std(value: &std::time::Duration) -> PolarsResult<i64> {
        i64::try_from(value.as_micros())
            .map_err(|_| out_of_range("df-derive: std::time::Duration value out of i64 us range"))
    }

    #[cfg(feature = "chrono")]
    #[inline(always)]
    fn from_instant<Tz: chrono::TimeZone>(value: &chrono::DateTime<Tz>) -> PolarsResult<i64> {
        Ok(value.timestamp_micros())
    }

    #[cfg(feature = "chrono")]
    #[inline(always)]
    fn from_naive(value: &chrono::NaiveDateTime) -> PolarsResult<i64> {
        Ok(value.and_utc().timestamp_micros())
    }

    #[cfg(feature = "chrono")]
    #[inline(always)]
    fn from_delta(value: &chrono::TimeDelta) -> PolarsResult<i64> {
        value
            .num_microseconds()
            .ok_or_else(|| out_of_range("df-derive: chrono::Duration value out of i64 us range"))
    }
}

impl Unit for Ns {
    const UNIT: TimeUnit = TimeUnit::Nanoseconds;

    #[inline(always)]
    fn from_std(value: &std::time::Duration) -> PolarsResult<i64> {
        i64::try_from(value.as_nanos())
            .map_err(|_| out_of_range("df-derive: std::time::Duration value out of i64 ns range"))
    }

    #[cfg(feature = "chrono")]
    #[inline(always)]
    fn from_instant<Tz: chrono::TimeZone>(value: &chrono::DateTime<Tz>) -> PolarsResult<i64> {
        value.timestamp_nanos_opt().ok_or_else(|| {
            out_of_range(
                "df-derive: DateTime<Tz> value is out of range for nanosecond timestamps (chrono supports approximately 1677..2262)",
            )
        })
    }

    #[cfg(feature = "chrono")]
    #[inline(always)]
    fn from_naive(value: &chrono::NaiveDateTime) -> PolarsResult<i64> {
        value.and_utc().timestamp_nanos_opt().ok_or_else(|| {
            out_of_range(
                "df-derive: NaiveDateTime value is out of range for nanosecond timestamps (chrono supports approximately 1677..2262)",
            )
        })
    }

    #[cfg(feature = "chrono")]
    #[inline(always)]
    fn from_delta(value: &chrono::TimeDelta) -> PolarsResult<i64> {
        value
            .num_nanoseconds()
            .ok_or_else(|| out_of_range("df-derive: chrono::Duration value out of i64 ns range"))
    }
}

// The builders in this file branch over an empty slot, where a number's reads
// a spare value instead (`primitive::stored`). Without the branch every empty
// slot converts a value, and irregular nulls took 13% to 37% longer
// (`24_irregular_nulls`).

/// Builder for `Datetime` and `Duration` columns in the unit `U`.
///
/// The value type decides whether a `Datetime` column has a time zone. A
/// `chrono::DateTime<Tz>` is an instant, so its column is
/// `Datetime(U, UTC)`: the value's own zone or offset is not kept, the
/// instant is. A `chrono::NaiveDateTime` names no zone and neither does its
/// column, `Datetime(U, None)`. Polars treats the two as different types.
pub struct TemporalBuilder<U>(Vec<i64>, PhantomData<U>);

/// How a temporal value maps onto [`TemporalBuilder`].
macro_rules! temporal {
    ($(#[$cfg:meta])? $ty:ty $(, $tz:ident)? => $logical:ident, $convert:ident, $default:ty) => {
        $(#[$cfg])?
        impl<U: Unit $(, $tz: chrono::TimeZone)?> ColumnBuilder<$ty> for TemporalBuilder<U> {
            fn schema(name: PlSmallStr, out: &mut Fields) {
                out.push((name, temporal!(@dtype $logical U)));
            }

            fn new() -> Self {
                Self(Vec::new(), PhantomData)
            }

            fn reserve(&mut self, additional: usize) {
                self.0.reserve(additional);
            }

            #[inline(always)]
            fn push(&mut self, value: Option<&$ty>) -> PolarsResult<()> {
                self.0.push(match value {
                    Some(value) => U::$convert(value)?,
                    None => 0,
                });
                Ok(())
            }

            #[inline(always)]
            fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
            where
                $ty: 'a,
                I: Iterator<Item = Option<&'a $ty>> + Clone,
            {
                try_extend(&mut self.0, values, U::$convert)
            }

            fn finish(
                self,
                name: PlSmallStr,
                validity: Option<&Bitmap>,
                out: &mut Vec<Column>,
            ) -> PolarsResult<()> {
                let values = Int64Chunked::with_chunk(name, primitive_array(self.0, validity)?);
                out.push(temporal!(@series $logical values U).into());
                Ok(())
            }
        }

        $(#[$cfg])?
        impl$(<$tz: chrono::TimeZone>)? Columnar for $ty {
            type Builder = TemporalBuilder<$default>;
        }
    };
    (@dtype Instant $u:ident) => {
        DataType::Datetime($u::UNIT, Some(polars::prelude::TimeZone::UTC))
    };
    (@dtype Datetime $u:ident) => { DataType::Datetime($u::UNIT, None) };
    (@dtype Duration $u:ident) => { DataType::Duration($u::UNIT) };
    (@series Instant $values:ident $u:ident) => {
        $values
            .into_datetime($u::UNIT, Some(polars::prelude::TimeZone::UTC))
            .into_series()
    };
    (@series Datetime $values:ident $u:ident) => {
        $values.into_datetime($u::UNIT, None).into_series()
    };
    (@series Duration $values:ident $u:ident) => {
        $values.into_duration($u::UNIT).into_series()
    };
}

// A point in time defaults to microseconds: the finest unit that holds every
// date `chrono` has, so the default never fails. Nanoseconds reach only from
// 1677 to 2262. A span of time is short, so it keeps every digit.
temporal!(std::time::Duration => Duration, from_std, Ns);
temporal!(#[cfg(feature = "chrono")] chrono::DateTime<Tz>, Tz => Instant, from_instant, Us);
temporal!(#[cfg(feature = "chrono")] chrono::NaiveDateTime => Datetime, from_naive, Us);
temporal!(#[cfg(feature = "chrono")] chrono::TimeDelta => Duration, from_delta, Ns);

#[cfg(feature = "chrono")]
pub use chrono_impls::{DateBuilder, TimeOfDayBuilder};

#[cfg(feature = "chrono")]
mod chrono_impls {
    use chrono::{Datelike as _, NaiveDate, NaiveTime, Timelike as _};
    use polars::prelude::{
        Column, DataType, Int32Chunked, Int64Chunked, IntoSeries, PlSmallStr, PolarsResult,
    };
    use polars_arrow::bitmap::Bitmap;

    use super::primitive_array;
    use crate::dataframe::{ColumnBuilder, Columnar, Fields};

    /// Builder for a `Date` column.
    ///
    /// The builder is part of the `chrono` feature, which is on by default.
    pub struct DateBuilder(Vec<i32>);

    /// Builder for a `Time` column (nanoseconds since midnight).
    ///
    /// The builder is part of the `chrono` feature, which is on by default.
    pub struct TimeOfDayBuilder(Vec<i64>);

    /// Days between 0001-01-01 (chrono's day 1) and the Unix epoch.
    const UNIX_EPOCH_DAY_FROM_CE: i32 = 719_163;

    #[inline(always)]
    fn days_since_epoch(date: &NaiveDate) -> i32 {
        date.num_days_from_ce() - UNIX_EPOCH_DAY_FROM_CE
    }

    #[inline(always)]
    fn nanos_since_midnight(time: &NaiveTime) -> i64 {
        i64::from(time.num_seconds_from_midnight()) * 1_000_000_000 + i64::from(time.nanosecond())
    }

    impl ColumnBuilder<NaiveDate> for DateBuilder {
        fn schema(name: PlSmallStr, out: &mut Fields) {
            out.push((name, DataType::Date));
        }

        fn new() -> Self {
            Self(Vec::new())
        }

        fn reserve(&mut self, additional: usize) {
            self.0.reserve(additional);
        }

        #[inline(always)]
        fn push(&mut self, value: Option<&NaiveDate>) -> PolarsResult<()> {
            self.0.push(value.map(days_since_epoch).unwrap_or_default());
            Ok(())
        }

        #[inline(always)]
        fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
        where
            I: Iterator<Item = Option<&'a NaiveDate>> + Clone,
        {
            self.0
                .extend(values.map(|value| value.map(days_since_epoch).unwrap_or_default()));
            Ok(())
        }

        fn finish(
            self,
            name: PlSmallStr,
            validity: Option<&Bitmap>,
            out: &mut Vec<Column>,
        ) -> PolarsResult<()> {
            let values = Int32Chunked::with_chunk(name, primitive_array(self.0, validity)?);
            out.push(values.into_date().into_series().into());
            Ok(())
        }
    }

    impl Columnar for NaiveDate {
        type Builder = DateBuilder;
    }

    impl ColumnBuilder<NaiveTime> for TimeOfDayBuilder {
        fn schema(name: PlSmallStr, out: &mut Fields) {
            out.push((name, DataType::Time));
        }

        fn new() -> Self {
            Self(Vec::new())
        }

        fn reserve(&mut self, additional: usize) {
            self.0.reserve(additional);
        }

        #[inline(always)]
        fn push(&mut self, value: Option<&NaiveTime>) -> PolarsResult<()> {
            self.0
                .push(value.map(nanos_since_midnight).unwrap_or_default());
            Ok(())
        }

        #[inline(always)]
        fn extend<'a, I>(&mut self, values: I) -> PolarsResult<()>
        where
            I: Iterator<Item = Option<&'a NaiveTime>> + Clone,
        {
            self.0
                .extend(values.map(|value| value.map(nanos_since_midnight).unwrap_or_default()));
            Ok(())
        }

        fn finish(
            self,
            name: PlSmallStr,
            validity: Option<&Bitmap>,
            out: &mut Vec<Column>,
        ) -> PolarsResult<()> {
            let values = Int64Chunked::with_chunk(name, primitive_array(self.0, validity)?);
            out.push(values.into_time().into_series().into());
            Ok(())
        }
    }

    impl Columnar for NaiveTime {
        type Builder = TimeOfDayBuilder;
    }
}
