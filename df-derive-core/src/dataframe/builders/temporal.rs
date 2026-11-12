use std::marker::PhantomData;

use polars::prelude::{
    Column, DataType, Int64Chunked, IntoSeries, PlSmallStr, PolarsError, PolarsResult, TimeUnit,
};
use polars_arrow::bitmap::Bitmap;

use super::primitive::primitive_array;
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

/// Builder for instants (`Datetime(U, None)`) and durations (`Duration(U)`).
pub struct TemporalBuilder<U>(Vec<i64>, PhantomData<U>);

/// How a temporal value maps onto [`TemporalBuilder`].
macro_rules! temporal {
    ($(#[$cfg:meta])? $ty:ty $(, $tz:ident)? => $logical:ident, $convert:ident, $default:ty) => {
        $(#[$cfg])?
        impl<U: Unit $(, $tz: chrono::TimeZone)?> ColumnBuilder<$ty> for TemporalBuilder<U> {
            fn schema(name: PlSmallStr, out: &mut Fields) {
                out.push((name, temporal!(@dtype $logical U)));
            }

            fn with_capacity(rows: usize) -> Self {
                Self(Vec::with_capacity(rows), PhantomData)
            }

            #[inline(always)]
            fn push(&mut self, value: &$ty) -> PolarsResult<()> {
                self.0.push(U::$convert(value)?);
                Ok(())
            }

            #[inline(always)]
            fn push_null(&mut self) {
                self.0.push(0);
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
    (@dtype Datetime $u:ident) => { DataType::Datetime($u::UNIT, None) };
    (@dtype Duration $u:ident) => { DataType::Duration($u::UNIT) };
    (@series Datetime $values:ident $u:ident) => {
        $values.into_datetime($u::UNIT, None).into_series()
    };
    (@series Duration $values:ident $u:ident) => {
        $values.into_duration($u::UNIT).into_series()
    };
}

temporal!(std::time::Duration => Duration, from_std, Ns);
temporal!(#[cfg(feature = "chrono")] chrono::DateTime<Tz>, Tz => Datetime, from_instant, Ms);
temporal!(#[cfg(feature = "chrono")] chrono::NaiveDateTime => Datetime, from_naive, Ms);
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
    pub struct DateBuilder(Vec<i32>);

    /// Builder for a `Time` column (nanoseconds since midnight).
    pub struct TimeOfDayBuilder(Vec<i64>);

    /// Days between 0001-01-01 (chrono's day 1) and the Unix epoch.
    const UNIX_EPOCH_DAY_FROM_CE: i32 = 719_163;

    impl ColumnBuilder<NaiveDate> for DateBuilder {
        fn schema(name: PlSmallStr, out: &mut Fields) {
            out.push((name, DataType::Date));
        }

        fn with_capacity(rows: usize) -> Self {
            Self(Vec::with_capacity(rows))
        }

        #[inline(always)]
        fn push(&mut self, value: &NaiveDate) -> PolarsResult<()> {
            self.0
                .push(value.num_days_from_ce() - UNIX_EPOCH_DAY_FROM_CE);
            Ok(())
        }

        #[inline(always)]
        fn push_null(&mut self) {
            self.0.push(0);
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

        fn with_capacity(rows: usize) -> Self {
            Self(Vec::with_capacity(rows))
        }

        #[inline(always)]
        fn push(&mut self, value: &NaiveTime) -> PolarsResult<()> {
            self.0.push(
                i64::from(value.num_seconds_from_midnight()) * 1_000_000_000
                    + i64::from(value.nanosecond()),
            );
            Ok(())
        }

        #[inline(always)]
        fn push_null(&mut self) {
            self.0.push(0);
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
