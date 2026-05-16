use crate::ir::{
    AccessChain, AccessStep, ColumnIR, ColumnSource, FieldIR, FieldSource, LeafShape, LeafSpec,
    ProjectionContext, TerminalLeafSpec, TupleElement, TupleProjectionStep, VecLayers,
    WrapperShape, column_name_for_ident,
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
        leaf_spec => columns.push(ColumnIR {
            name,
            source: ColumnSource::Field(root),
            leaf_spec: terminal_leaf(leaf_spec),
            wrapper_shape: field.wrapper_shape,
        }),
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
            project_tuple_elements(
                columns,
                root,
                &name,
                &WrapperShape::Leaf(LeafShape::Bare),
                inner,
                &path,
            );
            continue;
        }

        let leaf_spec = terminal_leaf(element.leaf_spec.clone());
        let (wrapper_shape, context) = compose_parent_with_element(parent_wrapper, element, step);
        columns.push(ColumnIR {
            name,
            source: ColumnSource::TupleProjection {
                root: root.clone(),
                path,
                context,
            },
            leaf_spec,
            wrapper_shape,
        });
    }
}

fn terminal_leaf(leaf: LeafSpec) -> TerminalLeafSpec {
    TerminalLeafSpec::new(leaf).expect("projection only emits terminal column leaves")
}

fn compose_parent_with_element(
    parent_wrapper: &WrapperShape,
    element: &TupleElement,
    terminal_step: TupleProjectionStep,
) -> (WrapperShape, ProjectionContext) {
    match parent_wrapper {
        WrapperShape::Leaf(LeafShape::Bare) => {
            (element.wrapper_shape.clone(), ProjectionContext::Static)
        }
        WrapperShape::Leaf(LeafShape::Optional { access, .. }) => (
            compose_option_with_element(&element.wrapper_shape),
            ProjectionContext::ParentOption {
                access: access.clone(),
            },
        ),
        WrapperShape::Vec(parent_layers) => (
            compose_vec_parent_with_element(parent_layers, &element.wrapper_shape),
            ProjectionContext::ParentVec {
                projection_layer: parent_layers.depth(),
                parent_inner_access: parent_layers.inner_access.clone(),
                terminal_step,
            },
        ),
    }
}

fn compose_option_with_element(element_shape: &WrapperShape) -> WrapperShape {
    match element_shape {
        WrapperShape::Leaf(LeafShape::Bare) => WrapperShape::Leaf(LeafShape::from_option_access(
            1,
            prepend_option_access(&AccessChain::empty()),
        )),
        WrapperShape::Leaf(LeafShape::Optional {
            option_layers,
            access,
        }) => WrapperShape::Leaf(LeafShape::from_option_access(
            1 + option_layers.get(),
            prepend_option_access(access),
        )),
        WrapperShape::Vec(layers) => {
            let mut new_layers = layers.layers.clone();
            new_layers[0].option_layers_above += 1;
            new_layers[0].access = prepend_option_access(&new_layers[0].access);
            WrapperShape::Vec(VecLayers {
                layers: new_layers,
                inner_option_layers: layers.inner_option_layers,
                inner_access: layers.inner_access.clone(),
            })
        }
    }
}

fn compose_vec_parent_with_element(
    parent_layers: &VecLayers,
    element_shape: &WrapperShape,
) -> WrapperShape {
    let mut composed_layers = parent_layers.layers.clone();
    let carried_inner_option = parent_layers.inner_option_layers;

    let composed_inner_option = match element_shape {
        WrapperShape::Vec(element_layers) => {
            let mut new_layers = element_layers.layers.clone();
            new_layers[0].option_layers_above += carried_inner_option;
            new_layers[0].access =
                prepend_parent_option_access(&parent_layers.inner_access, &new_layers[0].access);
            composed_layers.extend(new_layers);
            element_layers.inner_option_layers
        }
        WrapperShape::Leaf(leaf_shape) => carried_inner_option + leaf_shape.option_layers(),
    };
    let composed_inner_access = match element_shape {
        WrapperShape::Vec(element_layers) => element_layers.inner_access.clone(),
        WrapperShape::Leaf(LeafShape::Bare) => parent_layers.inner_access.clone(),
        WrapperShape::Leaf(LeafShape::Optional { access, .. }) => {
            concat_access_chains(&parent_layers.inner_access, access)
        }
    };

    WrapperShape::Vec(VecLayers {
        layers: composed_layers,
        inner_option_layers: composed_inner_option,
        inner_access: composed_inner_access,
    })
}

fn prepend_option_access(access: &AccessChain) -> AccessChain {
    let mut steps = Vec::with_capacity(access.steps.len() + 1);
    steps.push(AccessStep::Option);
    steps.extend(access.steps.iter().copied());
    AccessChain { steps }
}

fn concat_access_chains(left: &AccessChain, right: &AccessChain) -> AccessChain {
    let mut steps = Vec::with_capacity(left.steps.len() + right.steps.len());
    steps.extend(left.steps.iter().copied());
    steps.extend(right.steps.iter().copied());
    AccessChain { steps }
}

fn prepend_parent_option_access(parent_access: &AccessChain, access: &AccessChain) -> AccessChain {
    if parent_access.option_layers() > 0 {
        prepend_option_access(access)
    } else {
        access.clone()
    }
}
