// These integration fixtures intentionally favor compact scenario names,
// long exhaustive cases, and direct assertions over production-style lint
// polish. Keep `just lint` focused on library behavior.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::items_after_statements,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::redundant_clone,
    clippy::redundant_closure_for_method_calls,
    clippy::semicolon_if_nothing_returned,
    clippy::similar_names,
    clippy::struct_field_names,
    clippy::too_many_lines,
    clippy::unnecessary_literal_bound
)]

#[path = "runtime/28-option-vec-struct-validity.rs"]
mod option_vec_struct_validity;

#[path = "runtime/30-vec-option-struct.rs"]
mod vec_option_struct;

#[path = "runtime/31-option-vec-option-struct.rs"]
mod option_vec_option_struct;

#[path = "runtime/32-vec-vec-struct.rs"]
mod vec_vec_struct;

#[path = "runtime/33-vec-option-vec-mid-stack.rs"]
mod vec_option_vec_mid_stack;

#[path = "runtime/36-as-binary-attribute.rs"]
mod as_binary_attribute;

#[path = "runtime/38-smart-pointers.rs"]
mod smart_pointers;

#[path = "runtime/39-tuple-fields.rs"]
mod tuple_fields;

#[path = "runtime/41-decimal-custom-backend-attr.rs"]
mod decimal_custom_backend_attr;

#[path = "runtime/45-borrowed-references.rs"]
mod borrowed_references;

#[path = "runtime/47-nonzero-numerics.rs"]
mod nonzero_numerics;

#[path = "runtime/48-qualified-type-paths.rs"]
mod qualified_type_paths;

#[path = "runtime/49-generic-shadowing-and-bounds.rs"]
mod generic_shadowing_and_bounds;

#[path = "runtime/50-qualified-custom-name-collisions.rs"]
mod qualified_custom_name_collisions;

#[path = "runtime/51-qualified-custom-wrapper-collisions.rs"]
mod qualified_custom_wrapper_collisions;

#[path = "runtime/52-skip-field.rs"]
mod skip_field;

#[path = "runtime/53-associated-type-paths.rs"]
mod associated_type_paths;

#[path = "runtime/54-raw-ident-column-names.rs"]
mod raw_ident_column_names;

#[path = "runtime/56-as-string-fmt-error.rs"]
mod as_string_fmt_error;

#[path = "runtime/57-mixed-row-behavior-lock.rs"]
mod mixed_row_behavior_lock;

#[path = "runtime/58-mixed-wrapper-shape-row.rs"]
mod mixed_wrapper_shape_row;

#[path = "runtime/59-tuple-vector-projection.rs"]
mod tuple_vector_projection;

#[path = "runtime/60-flatten-field.rs"]
mod flatten_field;

#[path = "runtime/61-declared-encodings-and-projections.rs"]
mod declared_encodings_and_projections;

#[path = "runtime/62-instants-and-naive-datetimes.rs"]
mod instants_and_naive_datetimes;

#[path = "runtime/63-lifetimes-of-nested-types.rs"]
mod lifetimes_of_nested_types;

#[path = "runtime/64-row-at-a-time.rs"]
mod row_at_a_time;

#[path = "runtime/65-rename.rs"]
mod rename;

#[path = "runtime/66-fixed-size-arrays.rs"]
mod fixed_size_arrays;

#[path = "runtime/67-floats-bit-for-bit.rs"]
mod floats_bit_for_bit;

#[cfg(feature = "categorical")]
#[path = "runtime/68-categorical.rs"]
mod categorical;
