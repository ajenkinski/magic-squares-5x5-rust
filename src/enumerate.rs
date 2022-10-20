use itertools::Itertools;
use std::collections::HashSet;

type SquareVal = u8;

/// Different types of square components
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
enum Comp {
    Row(usize),
    Col(usize),
    MainDiag,
    MinorDiag,
}

type Coord = (usize, usize);
type SquareVec = Vec<SquareVal>;

fn component_indices(comp: Comp) -> Vec<Coord> {
    match comp {
        Comp::Row(r) => (0..5).map(|c| (r, c)).collect(),
        Comp::Col(c) => (0..5).map(|r| (r, c)).collect(),
        Comp::MainDiag => (0..5).map(|i| (i, i)).collect(),
        Comp::MinorDiag => (0..5).map(|r| (r, 4 - r)).collect(),
    }
}

#[test]
fn test_component_indices() {
    assert_eq!(
        component_indices(Comp::Row(1)),
        vec![(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    );
    assert_eq!(
        component_indices(Comp::Col(2)),
        vec![(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    );
    assert_eq!(
        component_indices(Comp::MainDiag),
        vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    );
    assert_eq!(
        component_indices(Comp::MinorDiag),
        vec![(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
    );
}

/// Holds pre-computed values used by generation algorithm
struct Env {
    /// Vec of vecs of coordinates, one vec for each row, column and diagonal
    all_component_coords: Vec<Vec<Coord>>,

    /// All possible valid rows, columns or diagonals, ignoring order
    all_vectors: Vec<SquareVec>,

    /// Index that allows looking up all vectors containing number x.  vectors_by_include[x - 1] is the set
    /// of indexes into all_vectors of vectors containing x.
    vectors_by_include: Vec<HashSet<usize>>,

    /// Index that allows looking up all vectors *not* containing number x.  vectors_by_exclude[x - 1] is the
    ///  set of indexes into all_vectors of vectors not containing x.
    vectors_by_exclude: Vec<HashSet<usize>>,
}

impl Env {
    fn new() -> Env {
        let mut all_component_coords: Vec<Vec<Coord>> = vec![];
        for i in 0..5 {
            all_component_coords.push(component_indices(Comp::Row(i)));
            all_component_coords.push(component_indices(Comp::Col(i)));
        }
        all_component_coords.push(component_indices(Comp::MainDiag));
        all_component_coords.push(component_indices(Comp::MinorDiag));

        let all_nums = (1 as SquareVal)..=25;
        let all_vectors: Vec<SquareVec> = all_nums.clone().combinations(5).collect();

        let mut vectors_by_include = vec![HashSet::<usize>::new(); 25];
        let mut vectors_by_exclude = vec![HashSet::<usize>::new(); 25];

        for (i, v) in all_vectors.iter().enumerate() {
            for x in v.iter() {
                vectors_by_include[(x - 1) as usize].insert(i);
            }

            for x in all_nums.clone() {
                if !v.contains(&x) {
                    vectors_by_exclude[(x - 1) as usize].insert(i);
                }
            }
        }

        Env {
            all_component_coords,
            all_vectors,
            vectors_by_include,
            vectors_by_exclude,
        }
    }

    fn filtered_vectors(&self, includes: &[SquareVal], excludes: &[SquareVal]) -> Vec<&SquareVec> {
        let vec_sets = includes
            .iter()
            .map(|i| &self.vectors_by_include[(i - 1) as usize])
            .chain(
                excludes
                    .iter()
                    .map(|i| &self.vectors_by_exclude[(i - 1) as usize]),
            )
            .collect_vec();

        if vec_sets.len() == 0 {
            vec![]
        } else {
            // intersection of vec_sets
            let vec_idxs = vec_sets
                .iter()
                .skip(1)
                .cloned()
                .fold(vec_sets[0].clone(), |s1, s2| &s1 & s2);

            vec_idxs.iter().map(|i| &self.all_vectors[*i]).collect()
        }
    }
}
