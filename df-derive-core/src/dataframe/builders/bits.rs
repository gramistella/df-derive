use polars_arrow::bitmap::Bitmap;

/// Multiplying eight bytes that are each 0 or 1 by this constant gathers
/// their low bits into the top byte of the product, first byte lowest. Byte
/// `i` sits at bit `8 * i` and the constant has bit `56 - 7 * i` set, so the
/// two meet at bit `56 + i`; no other pair of bits lands on the same place,
/// so nothing carries.
const GATHER: u64 = 0x0102_0408_1020_4080;

/// Packs one flag per byte into one flag per bit.
///
/// Flags are kept as bytes while a column is built, which makes appending
/// them a plain vector push, and packed here once at the end, eight at a
/// time.
pub(super) fn pack(flags: &[bool]) -> Bitmap {
    let (words, rest) = flags.as_chunks::<8>();
    let mut bytes = Vec::with_capacity(flags.len().div_ceil(8));
    bytes.extend(words.iter().map(|word| {
        let word = u64::from_le_bytes(word.map(u8::from));
        word.wrapping_mul(GATHER).to_be_bytes()[0]
    }));
    if !rest.is_empty() {
        let last = rest
            .iter()
            .enumerate()
            .fold(0, |byte, (bit, &flag)| byte | (u8::from(flag) << bit));
        bytes.push(last);
    }
    Bitmap::from_u8_vec(bytes, flags.len())
}

#[cfg(test)]
mod tests {
    use super::pack;

    /// Checks every length around the word boundaries, and every bit pattern
    /// within a word, against the definition: bit `i` is flag `i`.
    #[test]
    fn packed_bits_are_the_flags() {
        let flags: Vec<bool> = (0..256usize)
            .flat_map(|pattern| (0..8).map(move |bit| pattern >> bit & 1 == 1))
            .chain((0..67usize).map(|index| index % 3 == 0))
            .collect();
        for len in (0..=40).chain([flags.len() - 1, flags.len()]) {
            let packed = pack(&flags[..len]);
            assert_eq!(packed.len(), len);
            for (index, &flag) in flags[..len].iter().enumerate() {
                assert_eq!(packed.get_bit(index), flag, "bit {index} of {len}");
            }
        }
    }
}
