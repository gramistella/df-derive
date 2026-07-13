use crate::ir::{
    AccessChain, ColumnIR, FieldIR, FieldSource, LeafShape, LeafSpec, TerminalLeafSpec,
    TupleElement, TupleProjectionPath, TupleProjectionStep, VecLayers, WrapperShape,
    column_name_for_ident,
};

pub fn project_fields_to_columns(fields: Vec<FieldIR>) -> Vec<ColumnIR> {
    let mut columns = Vec::new();
    for field in fields {
        project_field(field, &mut columns);
    }
    columns
}

fn project_field(field: FieldIR, columns: &mut Vec<ColumnIR>) {
    let root = FieldSource {
        name: field.name.clone(),
        field_index: field.field_index,
        outer_smart_ptr_depth: field.outer_smart_ptr_depth,
    };
    let name = column_name_for_ident(&field.name);
    match field.leaf_spec {
        LeafSpec::Tuple(elements) => {
            project_tuple_elements(columns, &root, &name, &field.wrapper_shape, &elements, &[]);
        }
        leaf_spec => columns.push(ColumnIR::field(
            name,
            root,
            terminal_leaf(leaf_spec),
            field.wrapper_shape,
            field.nested_name_policy,
        )),
    }
}

fn project_tuple_elements(
    columns: &mut Vec<ColumnIR>,
    root: &FieldSource,
    column_prefix: &str,
    parent_wrapper: &WrapperShape,
    elements: &[TupleElement],
    path_prefix: &[TupleProjectionStep],
) {
    for (index, element) in elements.iter().enumerate() {
        let mut path = path_prefix.to_owned();
        let step = TupleProjectionStep {
            index,
            outer_smart_ptr_depth: element.outer_smart_ptr_depth,
        };
        path.push(step);
        let name = format!("{column_prefix}.field_{index}");
        if let LeafSpec::Tuple(inner) = &element.leaf_spec {
            debug_assert!(
                !matches!(parent_wrapper, WrapperShape::Vec(_)),
                "validation must reject nested tuple projections inside Vec tuple parents"
            );
            project_tuple_elements(
                columns,
                root,
                &name,
                &WrapperShape::Leaf(LeafShape::bare()),
                inner,
                &path,
            );
            continue;
        }

        let leaf_spec = terminal_leaf(element.leaf_spec.clone());
        let context = compose_parent_with_element(parent_wrapper, element);
        columns.push(match context {
            ProjectedColumnContext::Static { wrapper_shape } => ColumnIR::tuple_static(
                name,
                root.clone(),
                projection_path(path),
                leaf_spec,
                wrapper_shape,
            ),
            ProjectedColumnContext::ParentOption {
                wrapper_shape,
                parent_access,
            } => ColumnIR::tuple_parent_option(
                name,
                root.clone(),
                projection_path(path),
                parent_access,
                leaf_spec,
                wrapper_shape,
            ),
            ProjectedColumnContext::ParentVec {
                wrapper_shape,
                projection_layer,
                parent_inner_access,
            } => {
                debug_assert_eq!(
                    path.len(),
                    1,
                    "Vec tuple projections must end at the tuple element"
                );
                ColumnIR::tuple_parent_vec(
                    name,
                    root.clone(),
                    step,
                    projection_layer,
                    parent_inner_access,
                    leaf_spec,
                    wrapper_shape,
                )
            }
        });
    }
}

fn terminal_leaf(leaf: LeafSpec) -> TerminalLeafSpec {
    TerminalLeafSpec::new(leaf).expect("projection only emits terminal column leaves")
}

fn projection_path(path: Vec<TupleProjectionStep>) -> TupleProjectionPath {
    TupleProjectionPath::from_vec(path).expect("tuple projection path is never empty")
}

enum ProjectedColumnContext {
    Static {
        wrapper_shape: WrapperShape,
    },
    ParentOption {
        wrapper_shape: WrapperShape,
        parent_access: AccessChain,
    },
    ParentVec {
        wrapper_shape: VecLayers,
        projection_layer: usize,
        parent_inner_access: AccessChain,
    },
}

fn compose_parent_with_element(
    parent_wrapper: &WrapperShape,
    element: &TupleElement,
) -> ProjectedColumnContext {
    match parent_wrapper {
        WrapperShape::Leaf(shape) if shape.is_bare() => ProjectedColumnContext::Static {
            wrapper_shape: element.wrapper_shape.clone(),
        },
        WrapperShape::Leaf(shape) => ProjectedColumnContext::ParentOption {
            wrapper_shape: compose_option_with_element(&element.wrapper_shape),
            parent_access: shape.access().clone(),
        },
        WrapperShape::Vec(parent_layers) => ProjectedColumnContext::ParentVec {
            wrapper_shape: compose_vec_parent_with_element(parent_layers, &element.wrapper_shape),
            projection_layer: parent_layers.depth(),
            parent_inner_access: parent_layers.inner_access.clone(),
        },
    }
}

fn compose_option_with_element(element_shape: &WrapperShape) -> WrapperShape {
    match element_shape {
        WrapperShape::Leaf(shape) => {
            WrapperShape::Leaf(LeafShape::from_access(shape.access().prepend_option()))
        }
        WrapperShape::Vec(layers) => {
            let mut new_layers = layers.layers.clone();
            new_layers[0].access = new_layers[0].access.prepend_option();
            WrapperShape::Vec(VecLayers {
                layers: new_layers,
                inner_access: layers.inner_access.clone(),
            })
        }
    }
}

fn compose_vec_parent_with_element(
    parent_layers: &VecLayers,
    element_shape: &WrapperShape,
) -> VecLayers {
    let mut composed_layers = parent_layers.layers.clone();

    let composed_inner_access = match element_shape {
        WrapperShape::Vec(element_layers) => {
            let mut new_layers = element_layers.layers.clone();
            if parent_layers.inner_access.has_option() {
                new_layers[0].access = new_layers[0].access.prepend_option();
            }
            composed_layers.extend(new_layers);
            element_layers.inner_access.clone()
        }
        WrapperShape::Leaf(shape) => parent_layers.inner_access.concat(shape.access()),
    };

    VecLayers {
        layers: composed_layers,
        inner_access: composed_inner_access,
    }
}
