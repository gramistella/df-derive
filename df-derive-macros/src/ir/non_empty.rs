use std::ops::{Index, IndexMut};

/// A non-empty sequence with the first element encoded in the type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NonEmpty<T> {
    first: T,
    rest: Vec<T>,
}

#[allow(clippy::len_without_is_empty)]
impl<T> NonEmpty<T> {
    pub const fn new(first: T, rest: Vec<T>) -> Self {
        Self { first, rest }
    }

    pub fn from_vec(values: Vec<T>) -> Option<Self> {
        let mut values = values.into_iter();
        let first = values.next()?;
        Some(Self::new(first, values.collect()))
    }

    pub const fn len(&self) -> usize {
        1 + self.rest.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        ::std::iter::once(&self.first).chain(self.rest.iter())
    }
}

impl<T> Index<usize> for NonEmpty<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index == 0 {
            &self.first
        } else {
            &self.rest[index - 1]
        }
    }
}

impl<T> IndexMut<usize> for NonEmpty<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index == 0 {
            &mut self.first
        } else {
            &mut self.rest[index - 1]
        }
    }
}

impl<T> Extend<T> for NonEmpty<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.rest.extend(iter);
    }
}

impl<T> IntoIterator for NonEmpty<T> {
    type Item = T;
    type IntoIter = ::std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let Self { first, rest } = self;
        let mut values = Vec::with_capacity(1 + rest.len());
        values.push(first);
        values.extend(rest);
        values.into_iter()
    }
}
