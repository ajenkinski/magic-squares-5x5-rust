use fixedbitset::FixedBitSet;
use itertools::Itertools;
use rayon::prelude::*;
use std::sync::mpsc::channel;
use std::{
    collections::HashMap,
    io::{self, Write},
    thread,
};

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
type Square = [[SquareVal; 5]; 5];

const EMPTY_SQUARE: Square = [[0; 5]; 5];

/// Returns the set of coordinates for given component
fn get_component_coords(comp: Comp) -> Vec<Coord> {
    match comp {
        Comp::Row(r) => (0..5).map(|c| (r, c)).collect(),
        Comp::Col(c) => (0..5).map(|r| (r, c)).collect(),
        Comp::MainDiag => (0..5).map(|i| (i, i)).collect(),
        Comp::MinorDiag => (0..5).map(|r| (r, 4 - r)).collect(),
    }
}

/// Holds pre-computed values used by generation algorithm
pub struct Env {
    component_coords: HashMap<Comp, Vec<Coord>>,

    /// Vec of vecs of coordinates, one vec for each row, column and diagonal
    all_component_coords: Vec<Vec<Coord>>,

    /// All possible valid rows, columns or diagonals, ignoring order
    all_vectors: Vec<SquareVec>,

    /// Index that allows looking up all vectors containing number x.  vectors_by_include[x - 1] is the set
    /// of indexes into all_vectors of vectors containing x.
    vectors_by_include: Vec<FixedBitSet>,

    /// Index that allows looking up all vectors *not* containing number x.  vectors_by_exclude[x - 1] is the
    ///  set of indexes into all_vectors of vectors not containing x.
    vectors_by_exclude: Vec<FixedBitSet>,
}

impl Env {
    pub fn new() -> Env {
        let mut component_coords = HashMap::<Comp, Vec<Coord>>::new();
        for i in 0..5 {
            component_coords.insert(Comp::Row(i), get_component_coords(Comp::Row(i)));
            component_coords.insert(Comp::Col(i), get_component_coords(Comp::Col(i)));
        }
        component_coords.insert(Comp::MainDiag, get_component_coords(Comp::MainDiag));
        component_coords.insert(Comp::MinorDiag, get_component_coords(Comp::MinorDiag));

        let all_component_coords = component_coords.values().cloned().collect_vec();

        let all_nums = (1 as SquareVal)..=25;
        let all_vectors: Vec<SquareVec> = all_nums
            .clone()
            .combinations(5)
            .filter(|v| v.iter().sum::<SquareVal>() == 65)
            .collect();

        let max_vec_id: usize = 1394;
        let mut vectors_by_include = vec![FixedBitSet::with_capacity(max_vec_id); 25];
        let mut vectors_by_exclude = vec![FixedBitSet::with_capacity(max_vec_id); 25];

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
            component_coords,
            all_component_coords,
            all_vectors,
            vectors_by_include,
            vectors_by_exclude,
        }
    }

    /// Returns all component vectors which contain includes, and don't contain excludes
    fn filtered_vectors(&self, includes: &[SquareVal], excludes: &[SquareVal]) -> Vec<&SquareVec> {
        assert!(
            includes.len() + excludes.len() > 0,
            "At least one of includes or excludes must be non-empty"
        );

        let vec_sets = includes
            .iter()
            .map(|i| &self.vectors_by_include[(i - 1) as usize])
            .chain(
                excludes
                    .iter()
                    .map(|i| &self.vectors_by_exclude[(i - 1) as usize]),
            )
            .collect_vec();

        let mut vec_idxs = vec_sets[0].clone();
        for &s in vec_sets[1..].into_iter() {
            vec_idxs.intersect_with(s);
        }

        vec_idxs.ones().map(|i| &self.all_vectors[i]).collect()
    }

    /// Returns true if square is a valid 5x5 magic square
    fn square_is_valid(&self, square: &Square) -> bool {
        self.all_component_coords
            .iter()
            .map(|comp_coords| comp_coords.iter().map(|(r, c)| square[*r][*c]).sum::<u8>())
            .all(|sum| sum == 65)
    }

    /// Assign a row, column or diagonal to a square, returning a copy
    fn assign_vector(&self, square: &Square, comp: Comp, values: &SquareVec) -> Square {
        let mut new_square = *square;
        for (i, (r, c)) in self.component_coords[&comp].iter().enumerate() {
            new_square[*r][*c] = values[i];
        }
        new_square
    }

    /// Returns [(assignedVal, idx)] for a component, indicating where the
    /// assigned values for this component are.  assignedVals are the assigned values in component,
    /// and idexs are the indices of the non-zeros (0..4) along the component.
    fn assigned_values(&self, square: &Square, comp: Comp) -> Vec<(SquareVal, usize)> {
        self.component_coords[&comp]
            .iter()
            .enumerate()
            .filter_map(|(i, &(r, c))| {
                let val = square[r][c];
                if val == 0 {
                    None
                } else {
                    Some((val, i))
                }
            })
            .collect()
    }

    /// Return all the assigned values of a square
    fn all_square_values(&self, square: &Square) -> Vec<SquareVal> {
        square
            .iter()
            .flatten()
            .cloned()
            .filter(|val| *val != 0)
            .collect()
    }

    /// Return a list of all permutations of a list, but with only certain elements allowed to move.
    /// For example vector_permutations([1,3,4], vec) will return a list of all permutations of vec
    /// resulting from permuting the elements with index (0-based) 1, 3 and 4, but elements 0 and 2
    /// won't be moved.
    fn vector_permuations(
        &self,
        to_move: &[usize],
        vector: &SquareVec,
    ) -> impl Iterator<Item = SquareVec> {
        let to_move: Vec<_> = to_move.into();
        let vector = vector.clone();

        let mut perms = to_move.clone().into_iter().permutations(to_move.len());

        std::iter::from_fn(move || {
            if let Some(perm) = perms.next() {
                let mut new_vec = vector.clone();
                for (orig_i, new_i) in to_move.iter().zip(perm) {
                    new_vec[new_i] = vector[*orig_i];
                }
                Some(new_vec)
            } else {
                None
            }
        })
    }

    /// align_to is a list (value, index) pairs. It is assumed that vector contains all the indicated
    /// values.  Returns a copy of vector, with values moved to the corresponding index values.
    fn align_vector(&self, align_to: &[(SquareVal, usize)], vector: &SquareVec) -> SquareVec {
        let mut new_vec = vector.clone();

        for (val, i) in align_to.iter() {
            if new_vec[*i] != *val {
                let old_pos = new_vec.iter().position(|v| *v == *val).unwrap();
                new_vec.swap(*i, old_pos);
            }
        }

        new_vec
    }
}

/// Returns an iterator over all squares resulting from filling in comp on square
fn perform_step<'a>(
    env: &'a Env,
    square: &Square,
    comp: Comp,
) -> impl Iterator<Item = Square> + 'a {
    let assigned = env.assigned_values(square, comp);
    let (assigned_vals, assigned_indices): (Vec<_>, Vec<_>) = assigned.iter().copied().unzip();

    let vals_to_exclude = env
        .all_square_values(square)
        .into_iter()
        .filter(|v| !assigned_vals.contains(v))
        .collect_vec();

    let square = square.clone();

    env.filtered_vectors(&assigned_vals, &vals_to_exclude)
        .into_iter()
        .flat_map(move |new_component_vec| {
            let to_move = (0usize..5)
                .filter(|i| !assigned_indices.contains(i))
                .collect_vec();
            let aligned_vec = env.align_vector(&assigned, new_component_vec);
            // This copy is to work around a problem with capture
            let comp = comp.clone();
            env.vector_permuations(&to_move, &aligned_vec)
                .map(move |vec| env.assign_vector(&square, comp, &vec))
        })
}

/// Given a starting square with only the main diagonal filled in, returns an iterator over all filled in squares
/// with the starting diagonal.
fn squares_for_main_diag<'a>(
    env: &'a Env,
    main_diag_square: &Square,
) -> impl Iterator<Item = Square> + 'a {
    // implement multi-step backtracking by applying flat_map to the result of each previous step

    // fill in minor diag
    let it = perform_step(env, main_diag_square, Comp::MinorDiag);

    // constraint I > B, H > B, G > F > B
    let it = it.filter(|square| {
        square[3][1] > square[0][0]
            && square[1][3] > square[0][0]
            && square[4][0] > square[0][4]
            && square[0][4] > square[0][0]
    });

    // fill in row 0
    let it = it.flat_map(|square| perform_step(env, &square, Comp::Row(0)));

    // fill column 1
    let it = it.flat_map(|square| perform_step(env, &square, Comp::Col(1)));

    // Fill column 3
    let it = it.flat_map(|square| perform_step(env, &square, Comp::Col(3)));

    // fill row 4
    let it = it.flat_map(|square| perform_step(env, &square, Comp::Row(4)));

    // fill row 2
    let it = it.flat_map(|square| perform_step(env, &square, Comp::Row(2)));

    // fill column 2
    let it = it.flat_map(|square| perform_step(env, &square, Comp::Col(2)));

    // fill column 0
    let it = it.flat_map(|square| perform_step(env, &square, Comp::Col(0)));

    // fill column 4
    let it = it.flat_map(|square| perform_step(env, &square, Comp::Col(4)));

    // finally, filter out invalid squares.  The algorithm ensures that most components are valid, but row 1 and row 3
    // could still be invalid.
    let it = it.filter(|square| env.square_is_valid(square));

    // For squares with a center value of less than 13, add the "inverse" square
    let it = it.flat_map(|square| {
        let with_inverse = if square[2][2] < 13 {
            vec![square, square.map(|row| row.map(|v| 26 - v))]
        } else {
            vec![square]
        };
        with_inverse.into_iter()
    });

    it
}

/// Returns an iterator of all squares with just the main diagonal filled in
fn main_diag_squares<'a>(env: &'a Env) -> impl Iterator<Item = Square> + 'a {
    let center_values = (1 as SquareVal)..=13;
    let center_squares = center_values.map(|center| {
        let mut square = EMPTY_SQUARE;
        square[2][2] = center;
        square
    });

    center_squares
        .flat_map(|center_square| perform_step(env, &center_square, Comp::MainDiag))
        .filter(|square| {
            // Contraint E > B, D > C > B
            square[4][4] > square[0][0]
                && square[3][3] > square[1][1]
                && square[1][1] > square[0][0]
        })
}

/// Returns an iterator over all "basic" 5x5 magic squares, computed serially
pub fn generate_all_squares<'a>(env: &'a Env) -> impl Iterator<Item = Square> + 'a {
    main_diag_squares(env).flat_map(|square| squares_for_main_diag(env, &square))
}

/// Generates all "basic" 5x5 magic squares in parallel.  Prints progress as it goes along, and returns the total
/// number of squares found.
pub fn generate_all_squares_parallel<'a>(env: &'a Env) -> usize {
    println!("Starting parallel computation");

    thread::scope(|scope| {
        let (sender, receiver) = channel();

        scope.spawn(|| {
            main_diag_squares(env)
                .par_bridge()
                .for_each_with(sender, |s, square| {
                    let count = squares_for_main_diag(env, &square).count();
                    s.send(count).unwrap();
                });
        });

        let mut num_squares = 0;
        for n in receiver.iter() {
            num_squares += n;
            if num_squares % 1000 == 0 {
                print!("Found {} squares\r", num_squares);
                io::stdout().flush().unwrap();
            }
        }

        num_squares
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_indices() {
        assert_eq!(
            get_component_coords(Comp::Row(1)),
            vec![(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
        );
        assert_eq!(
            get_component_coords(Comp::Col(2)),
            vec![(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
        );
        assert_eq!(
            get_component_coords(Comp::MainDiag),
            vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        );
        assert_eq!(
            get_component_coords(Comp::MinorDiag),
            vec![(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
        );
    }

    #[test]
    fn test_new_env() {
        let env = Env::new();
        assert_eq!(env.all_component_coords.len(), 12);
        assert!(env.all_component_coords.iter().all(|cc| cc.len() == 5));

        assert_eq!(env.all_vectors.len(), 1394);
        assert!(env.all_vectors.iter().all(|v| v.len() == 5));
        assert!(env
            .all_vectors
            .iter()
            .all(|v| v.iter().sum::<SquareVal>() == 65));

        for (i, vi) in env.vectors_by_include.iter().enumerate() {
            for vec_idx in vi.ones() {
                assert!(env.all_vectors[vec_idx].contains(&((i + 1) as u8)));
            }
        }

        for (i, ve) in env.vectors_by_exclude.iter().enumerate() {
            for vec_idx in ve.ones() {
                assert!(!env.all_vectors[vec_idx].contains(&((i + 1) as u8)));
            }
        }
    }

    #[test]
    fn test_filtered_vectors() {
        let env = Env::new();
        let vecs = env.filtered_vectors(&[1, 23], &[]);
        assert!(vecs.len() > 0);
        for v in vecs {
            assert!(v.contains(&1));
            assert!(v.contains(&23));
        }

        let vecs = env.filtered_vectors(&[1, 23], &[2]);
        assert!(vecs.len() > 0);
        for v in vecs {
            assert!(v.contains(&1));
            assert!(v.contains(&23));
            assert!(!v.contains(&2));
        }
    }

    #[test]
    fn test_square_is_valid() {
        let env = Env::new();
        assert!(!env.square_is_valid(&EMPTY_SQUARE));

        let valid_square: Square = [
            [1, 22, 21, 18, 3],
            [20, 2, 15, 9, 19],
            [14, 16, 13, 10, 12],
            [7, 17, 11, 24, 6],
            [23, 8, 5, 4, 25],
        ];

        assert!(env.square_is_valid(&valid_square));
    }

    #[test]
    fn test_assign_vector() {
        let env = Env::new();

        let expected = [
            [0, 0, 0, 0, 0],
            [20, 2, 15, 9, 19],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ];
        let actual = env.assign_vector(&EMPTY_SQUARE, Comp::Row(1), &vec![20, 2, 15, 9, 19]);
        assert_eq!(expected, actual);

        let expected = [
            [0, 0, 1, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 4, 0, 0],
            [0, 0, 5, 0, 0],
        ];
        let actual = env.assign_vector(&EMPTY_SQUARE, Comp::Col(2), &vec![1, 2, 3, 4, 5]);
        assert_eq!(expected, actual);

        let expected = [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ];
        let actual = env.assign_vector(&EMPTY_SQUARE, Comp::MainDiag, &vec![1, 2, 3, 4, 5]);
        assert_eq!(expected, actual);

        let expected = [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 2, 0],
            [0, 0, 3, 0, 0],
            [0, 4, 0, 0, 0],
            [5, 0, 0, 0, 0],
        ];
        let actual = env.assign_vector(&EMPTY_SQUARE, Comp::MinorDiag, &vec![1, 2, 3, 4, 5]);
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_assigned_values() {
        let env = Env::new();

        assert_eq!(env.assigned_values(&EMPTY_SQUARE, Comp::Row(1)), vec![]);

        let square = [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 2, 0],
            [0, 0, 3, 0, 0],
            [0, 4, 0, 0, 0],
            [5, 0, 0, 0, 0],
        ];

        assert_eq!(env.assigned_values(&square, Comp::Row(0)), vec![(1, 4)]);
        assert_eq!(env.assigned_values(&square, Comp::Col(1)), vec![(4, 3)]);
        assert_eq!(env.assigned_values(&square, Comp::MainDiag), vec![(3, 2)]);
        assert_eq!(
            env.assigned_values(&square, Comp::MinorDiag),
            vec![(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)]
        );
    }

    #[test]
    fn test_all_square_values() {
        let env = Env::new();

        assert_eq!(env.all_square_values(&EMPTY_SQUARE), vec![]);

        let square = [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 2, 0],
            [0, 0, 3, 0, 0],
            [0, 4, 0, 0, 0],
            [5, 0, 0, 0, 0],
        ];

        assert_eq!(env.all_square_values(&square), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_vector_permutations() {
        let env = Env::new();

        let expected = vec![vec![1, 2, 3, 4, 5], vec![1, 2, 5, 4, 3]];
        let actual = env
            .vector_permuations(&[2, 4], &vec![1, 2, 3, 4, 5])
            .collect_vec();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_align_vector() {
        let env = Env::new();

        let aligned = env.align_vector(&[(1, 3), (4, 0)], &vec![1, 2, 3, 4, 5]);
        assert_eq!(aligned[0], 4);
        assert_eq!(aligned[3], 1);
        assert_eq!(
            aligned.iter().sorted().cloned().collect_vec(),
            vec![1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn test_perform_step() {
        let env = Env::new();

        let mut square = EMPTY_SQUARE;
        square[2][2] = 1;

        let num_vecs = env.filtered_vectors(&[1], &[]).len();
        // There should be 24 permutions of each vector, if only one value isn't allowed to move
        let expected_num_next_squares = num_vecs * 24;

        assert_eq!(
            perform_step(&env, &square, Comp::MainDiag).count(),
            expected_num_next_squares
        );

        let square = [
            [2, 0, 0, 0, 0],
            [0, 13, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 24, 0],
            [0, 0, 0, 0, 25],
        ];
        // taken from haskell version, which I believe to be correct
        let expected_num_next_squares: usize = 2448;
        assert_eq!(
            perform_step(&env, &square, Comp::MinorDiag).count(),
            expected_num_next_squares
        );
    }

    #[test]
    fn test_main_diag_squares() {
        let env = Env::new();
        assert_eq!(main_diag_squares(&env).count(), 10908);
    }

    #[test]
    fn test_squares_for_main_diag() {
        let env = Env::new();
        let square = [
            [2, 0, 0, 0, 0],
            [0, 13, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 24, 0],
            [0, 0, 0, 0, 25],
        ];

        let filled = squares_for_main_diag(&env, &square);
        assert!(filled.take(10).all(|square| env.square_is_valid(&square)));
    }
}
