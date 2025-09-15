// Test handling of self-referential and cyclic structures

use df_derive::ToDataFrame;
#[path = "../common.rs"]
mod core;
use crate::core::dataframe::ToDataFrame;
use crate::core::dataframe::ToDataFrameVec;

// Simple self-referential structure (tree-like)
#[derive(ToDataFrame)]
struct TreeNode {
    id: u32,
    value: String,
    // Note: Direct self-reference would create infinite recursion
    // So we use Option<Box<T>> pattern or just the data without children
    parent_id: Option<u32>,
    depth: u32,
}

// Mutual reference structures
#[derive(Clone, ToDataFrame)]
struct Person {
    id: u32,
    name: String,
    age: u32,
    // Instead of direct Company reference, use ID
    company_id: Option<u32>,
}

#[derive(Clone, ToDataFrame)]
struct Company {
    id: u32,
    name: String,
    founded: i32,
    // Instead of Vec<Person>, use a separate structure
    employee_count: u32,
}

// Test structure with references to other types by ID
#[derive(ToDataFrame)]
struct PersonCompanyRelation {
    person: Person,
    company: Option<Company>,
}

// Test with Vec of self-similar structures
#[derive(ToDataFrame)]
struct Category {
    id: u32,
    name: String,
    parent_category_id: Option<u32>,
    level: u32,
}

#[derive(ToDataFrame)]
struct CategoryHierarchy {
    root_category: String,
    categories: Vec<Category>,
}

// Test forward references (where struct A references struct B defined later)
#[derive(ToDataFrame)]
struct ForwardRefA {
    id: u32,
    name: String,
    // This references ForwardRefB which is defined later
    b_data: Option<ForwardRefB>,
}

#[derive(ToDataFrame)]
struct ForwardRefB {
    value: f64,
    description: String,
}

fn main() {
    test_tree_node_structure();
    test_person_company_relation();
    test_category_hierarchy();
    test_forward_references();
    test_empty_structures_with_references();
}

fn test_tree_node_structure() {
    let root = TreeNode {
        id: 1,
        value: "Root".to_string(),
        parent_id: None,
        depth: 0,
    };

    let child = TreeNode {
        id: 2,
        value: "Child".to_string(),
        parent_id: Some(1),
        depth: 1,
    };

    let grandchild = TreeNode {
        id: 3,
        value: "Grandchild".to_string(),
        parent_id: Some(2),
        depth: 2,
    };

    // Test individual nodes
    let root_df = root.to_dataframe().unwrap();
    assert_eq!(root_df.shape(), (1, 4));

    let child_df = child.to_dataframe().unwrap();
    assert_eq!(child_df.shape(), (1, 4));

    // Test vector of nodes (simulating a flattened tree)
    let tree_nodes = vec![root, child, grandchild];
    let tree_df = tree_nodes.to_dataframe().unwrap();
    assert_eq!(tree_df.shape(), (3, 4));

    let expected_columns = ["id", "value", "parent_id", "depth"];
    for expected in &expected_columns {
        assert!(
            tree_df
                .get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Expected column '{}' not found",
            expected
        );
    }

    println!("✅ Tree node structure test passed!");
}

fn test_person_company_relation() {
    let person = Person {
        id: 1,
        name: "John Doe".to_string(),
        age: 30,
        company_id: Some(100),
    };

    let company = Company {
        id: 100,
        name: "Tech Corp".to_string(),
        founded: 2010,
        employee_count: 250,
    };

    let relation = PersonCompanyRelation {
        person: person.clone(),
        company: Some(company.clone()),
    };

    let relation_df = relation.to_dataframe().unwrap();

    // Should have flattened: person.id, person.name, person.age, person.company_id,
    // company.id, company.name, company.founded, company.employee_count
    assert_eq!(relation_df.shape(), (1, 8));

    let expected_columns = [
        "person.id",
        "person.name",
        "person.age",
        "person.company_id",
        "company.id",
        "company.name",
        "company.founded",
        "company.employee_count",
    ];

    for expected in &expected_columns {
        assert!(
            relation_df
                .get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Expected column '{}' not found",
            expected
        );
    }

    // Test with None company
    let relation_none = PersonCompanyRelation {
        person: person.clone(),
        company: None,
    };

    let relation_none_df = relation_none.to_dataframe().unwrap();
    assert_eq!(relation_none_df.shape(), (1, 8));
    assert_eq!(
        relation_none_df.get_column_names(),
        relation_df.get_column_names()
    );

    println!("✅ Person-Company relation test passed!");
}

fn test_category_hierarchy() {
    let hierarchy = CategoryHierarchy {
        root_category: "Electronics".to_string(),
        categories: vec![
            Category {
                id: 1,
                name: "Electronics".to_string(),
                parent_category_id: None,
                level: 0,
            },
            Category {
                id: 2,
                name: "Computers".to_string(),
                parent_category_id: Some(1),
                level: 1,
            },
            Category {
                id: 3,
                name: "Laptops".to_string(),
                parent_category_id: Some(2),
                level: 2,
            },
            Category {
                id: 4,
                name: "Gaming Laptops".to_string(),
                parent_category_id: Some(3),
                level: 3,
            },
        ],
    };

    let df = hierarchy.to_dataframe().unwrap();

    // Should have: root_category, categories.id, categories.name,
    // categories.parent_category_id, categories.level
    assert_eq!(df.shape(), (1, 5));

    let expected_columns = [
        "root_category",
        "categories.id",
        "categories.name",
        "categories.parent_category_id",
        "categories.level",
    ];

    for expected in &expected_columns {
        assert!(
            df.get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Expected column '{}' not found",
            expected
        );
    }

    println!("✅ Category hierarchy test passed!");
}

fn test_forward_references() {
    let forward_ref = ForwardRefA {
        id: 1,
        name: "Forward Test".to_string(),
        b_data: Some(ForwardRefB {
            value: 42.0,
            description: "Forward referenced data".to_string(),
        }),
    };

    let df = forward_ref.to_dataframe().unwrap();

    // Should have: id, name, b_data.value, b_data.description
    assert_eq!(df.shape(), (1, 4));

    let expected_columns = ["id", "name", "b_data.value", "b_data.description"];
    for expected in &expected_columns {
        assert!(
            df.get_column_names()
                .iter()
                .any(|col| col.as_str() == *expected),
            "Expected column '{}' not found",
            expected
        );
    }

    // Test with None b_data
    let forward_ref_none = ForwardRefA {
        id: 2,
        name: "No Forward Data".to_string(),
        b_data: None,
    };

    let df_none = forward_ref_none.to_dataframe().unwrap();
    assert_eq!(df_none.shape(), (1, 4));
    assert_eq!(df_none.get_column_names(), df.get_column_names());

    println!("✅ Forward references test passed!");
}

fn test_empty_structures_with_references() {
    // Test empty vectors and None values in self-referential structures
    let empty_hierarchy = CategoryHierarchy {
        root_category: "Empty".to_string(),
        categories: vec![],
    };

    let df = empty_hierarchy.to_dataframe().unwrap();
    assert_eq!(df.shape(), (1, 5));

    // Test empty dataframes
    let empty_tree_df = TreeNode::empty_dataframe().unwrap();
    assert_eq!(empty_tree_df.shape(), (0, 4));

    let empty_relation_df = PersonCompanyRelation::empty_dataframe().unwrap();
    assert_eq!(empty_relation_df.shape(), (0, 8));

    let empty_hierarchy_df = CategoryHierarchy::empty_dataframe().unwrap();
    assert_eq!(empty_hierarchy_df.shape(), (0, 5));

    println!("✅ Empty structures with references test passed!");
}
