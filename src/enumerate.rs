use fixedbitset::FixedBitSet;
use itertools::Itertools;
use rayon::prelude::{ParallelBridge, ParallelIterator};
use std::{
    collections::HashMap,
    io::{Read, Write},
};

// Length of a side
const N: usize = 5;
// num cells in a square
const N_CELLS: usize = N * N;

/// A value between 1 and 25
type SquareVal = u8;

const MAX_VAL: SquareVal = (N * N) as SquareVal;

/// Expected sum of row, column or diagonal
const COMP_SUM: SquareVal = 65;

/// Different types of square components
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
enum Comp {
    Row(usize),
    Col(usize),
    MainDiag,
    MinorDiag,
}

impl Comp {
    /// Returns the set of coordinates for given component
    fn coords(&self) -> Vec<Coord> {
        match self {
            Comp::Row(r) => (0..N).map(|c| (*r, c)).collect(),
            Comp::Col(c) => (0..N).map(|r| (r, *c)).collect(),
            Comp::MainDiag => (0..N).map(|i| (i, i)).collect(),
            Comp::MinorDiag => (0..N).map(|r| (r, N - r - 1)).collect(),
        }
    }
}

/// A (row, column) coordinate of a square, 0 based
type Coord = (usize, usize);

/// A length-5 vector of values representing a possible row, column or diagonal of a square
type SquareVec = [SquareVal; N];

/// A magic square represented as a NxN array
type Square = [SquareVec; N];

/// An empty square constant.  0 is used to represent an un-filled-in value
const EMPTY_SQUARE: Square = [[0; N]; N];

/// Convert a length-N Vec to a SquareVec
fn vec_into_square_vec(vec: Vec<SquareVal>) -> SquareVec {
    vec.try_into().unwrap()
}

/// align_to is a list (value, index) pairs. It is assumed that vector contains all the indicated
/// values.  Returns a copy of vector, with values moved to the corresponding index values.
fn align_vector(align_to: &[(SquareVal, usize)], vector: &SquareVec) -> SquareVec {
    let mut new_vec = *vector;

    for &(val, i) in align_to.iter() {
        if new_vec[i] != val {
            let old_pos = new_vec.iter().position(|v| *v == val).unwrap();
            new_vec.swap(i, old_pos);
        }
    }

    new_vec
}

/// Return a list of all permutations of a list, but with only certain elements allowed to move.
/// For example vector_permutations([1,3,4], vec) will return a list of all permutations of vec
/// resulting from permuting the elements with index (0-based) 1, 3 and 4, but elements 0 and 2
/// won't be moved.
fn vector_permutations(to_move: &[usize], vector: &SquareVec) -> impl Iterator<Item = SquareVec> {
    let to_move: Vec<_> = to_move.into();
    let vector = *vector;

    to_move
        .clone()
        .into_iter()
        .permutations(to_move.len())
        .map(move |perm| {
            let mut new_vec = vector;
            for (orig_i, new_i) in to_move.iter().zip(perm) {
                new_vec[new_i] = vector[*orig_i];
            }
            new_vec
        })
}

/// Return all the assigned values of a square
fn all_square_values(square: &Square) -> impl Iterator<Item = SquareVal> + '_ {
    square.iter().flatten().copied().filter(|val| *val != 0)
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
    /// set of indexes into all_vectors of vectors not containing x.
    vectors_by_exclude: Vec<FixedBitSet>,
}

impl Env {
    /// Initialize a new Env struct with precomputed values to use in computations
    pub fn new() -> Env {
        let all_components = (0..N)
            .flat_map(|i| [Comp::Row(i), Comp::Col(i)])
            .chain([Comp::MainDiag, Comp::MinorDiag]);

        let component_coords: HashMap<Comp, Vec<(usize, usize)>> =
            all_components.map(|comp| (comp, comp.coords())).collect();

        let all_component_coords = component_coords.values().cloned().collect_vec();

        let all_nums = 1..=MAX_VAL;
        let all_vectors: Vec<SquareVec> = all_nums
            .clone()
            .combinations(N)
            .filter(|v| v.iter().sum::<SquareVal>() == COMP_SUM)
            .map(vec_into_square_vec)
            .collect();

        let num_vecs = all_vectors.len();
        let mut vectors_by_include = vec![FixedBitSet::with_capacity(num_vecs); N_CELLS];
        let mut vectors_by_exclude = vec![FixedBitSet::with_capacity(num_vecs); N_CELLS];

        for (i, v) in all_vectors.iter().enumerate() {
            for x in v {
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

        // intersection of all sets
        let mut vec_idxs = vec_sets[0].clone();
        for &s in &vec_sets[1..] {
            vec_idxs.intersect_with(s);
        }

        vec_idxs.ones().map(|i| &self.all_vectors[i]).collect()
    }

    /// Returns true if square is a valid NxN magic square
    fn square_is_valid(&self, square: &Square) -> bool {
        self.all_component_coords
            .iter()
            .map(|comp_coords| {
                comp_coords
                    .iter()
                    .map(|(r, c)| square[*r][*c])
                    .sum::<SquareVal>()
            })
            .all(|sum| sum == COMP_SUM)
    }

    /// Assign a row, column or diagonal to a square, returning a copy
    fn assign_vector(&self, square: &Square, comp: Comp, values: &SquareVec) -> Square {
        let mut new_square = *square;
        for (i, &(r, c)) in self.component_coords[&comp].iter().enumerate() {
            new_square[r][c] = values[i];
        }
        new_square
    }

    /// Returns `[(assignedVal, idx)]` for a component, indicating where the
    /// assigned values for this component are.  assignedVals are the assigned values in component,
    /// and idxs are the indices of the non-zeros (0..4) along the component.
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
}

impl Default for Env {
    fn default() -> Self {
        Self::new()
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

    let vals_to_exclude = all_square_values(square)
        .filter(|v| !assigned_vals.contains(v))
        .collect_vec();

    let square = *square;

    env.filtered_vectors(&assigned_vals, &vals_to_exclude)
        .into_iter()
        .flat_map(move |new_component_vec| {
            let to_move = (0usize..N)
                .filter(|i| !assigned_indices.contains(i))
                .collect_vec();
            let aligned_vec = align_vector(&assigned, new_component_vec);

            vector_permutations(&to_move, &aligned_vec)
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
    perform_step(env, main_diag_square, Comp::MinorDiag)
        // constraint I > B, H > B, G > F > B
        .filter(|square| {
            square[3][1] > square[0][0]
                && square[1][3] > square[0][0]
                && square[4][0] > square[0][4]
                && square[0][4] > square[0][0]
        })
        // fill in row 0
        .flat_map(|square| perform_step(env, &square, Comp::Row(0)))
        // fill column 1
        .flat_map(|square| perform_step(env, &square, Comp::Col(1)))
        // Fill column 3
        .flat_map(|square| perform_step(env, &square, Comp::Col(3)))
        // fill row 4
        .flat_map(|square| perform_step(env, &square, Comp::Row(4)))
        // fill row 2
        .flat_map(|square| perform_step(env, &square, Comp::Row(2)))
        // fill column 2
        .flat_map(|square| perform_step(env, &square, Comp::Col(2)))
        // fill column 0
        .flat_map(|square| perform_step(env, &square, Comp::Col(0)))
        // fill column 4
        .flat_map(|square| perform_step(env, &square, Comp::Col(4)))
        // finally, filter out invalid squares.  The algorithm ensures that most components are valid, but row 1 and row 3
        // could still be invalid.
        .filter(|square| env.square_is_valid(square))
        // For squares with a center value of less than 13, add the "inverse" square
        .flat_map(|square| {
            if square[2][2] < 13 {
                vec![square, square.map(|row| row.map(|v| 26 - v))]
            } else {
                vec![square]
            }
        })
}

/// Returns an iterator of all squares with just the main diagonal filled in
fn main_diag_squares(env: &Env) -> impl Iterator<Item = Square> + '_ {
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

/// Returns an iterator over all "basic" NxN magic squares, computed serially
pub fn generate_all_squares(env: &Env) -> impl Iterator<Item = Square> + '_ {
    main_diag_squares(env).flat_map(|square| squares_for_main_diag(env, &square))
}

/// Returns a parallel iterator over all "basic" NxN magic squares, computed in parallel
pub fn generate_all_squares_parallel(env: &Env) -> impl ParallelIterator<Item = Square> + '_ {
    main_diag_squares(env)
        .par_bridge()
        .flat_map_iter(|square| squares_for_main_diag(env, &square))
}

/// Write a square in binary format
pub fn write_square(square: &Square, writer: &mut impl Write) -> std::io::Result<()> {
    let buf = square.iter().flatten().copied().collect_vec();
    writer.write_all(buf.as_slice())
}

/// Read a square from a reader, in the format written out by write_square
pub fn read_square(reader: &mut impl Read) -> std::io::Result<Square> {
    let mut buf = [0u8; N_CELLS];
    reader.read_exact(&mut buf)?;

    let mut square = EMPTY_SQUARE;
    for r in 0..N {
        for c in 0..N {
            square[r][c] = buf[r * N + c];
        }
    }

    Ok(square)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_indices() {
        assert_eq!(
            Comp::Row(1).coords(),
            vec![(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
        );
        assert_eq!(
            Comp::Col(2).coords(),
            vec![(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
        );
        assert_eq!(
            Comp::MainDiag.coords(),
            vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        );
        assert_eq!(
            Comp::MinorDiag.coords(),
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
            .all(|v| v.iter().sum::<SquareVal>() == COMP_SUM));

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
        assert!(!vecs.is_empty());
        for v in vecs {
            assert!(v.contains(&1));
            assert!(v.contains(&23));
        }

        let vecs = env.filtered_vectors(&[1, 23], &[2]);
        assert!(!vecs.is_empty());
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
        let actual = env.assign_vector(&EMPTY_SQUARE, Comp::Row(1), &[20, 2, 15, 9, 19]);
        assert_eq!(expected, actual);

        let expected = [
            [0, 0, 1, 0, 0],
            [0, 0, 2, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 4, 0, 0],
            [0, 0, 5, 0, 0],
        ];
        let actual = env.assign_vector(&EMPTY_SQUARE, Comp::Col(2), &[1, 2, 3, 4, 5]);
        assert_eq!(expected, actual);

        let expected = [
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 4, 0],
            [0, 0, 0, 0, 5],
        ];
        let actual = env.assign_vector(&EMPTY_SQUARE, Comp::MainDiag, &[1, 2, 3, 4, 5]);
        assert_eq!(expected, actual);

        let expected = [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 2, 0],
            [0, 0, 3, 0, 0],
            [0, 4, 0, 0, 0],
            [5, 0, 0, 0, 0],
        ];
        let actual = env.assign_vector(&EMPTY_SQUARE, Comp::MinorDiag, &[1, 2, 3, 4, 5]);
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
        assert_eq!(all_square_values(&EMPTY_SQUARE).collect_vec(), vec![]);

        let square = [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 2, 0],
            [0, 0, 3, 0, 0],
            [0, 4, 0, 0, 0],
            [5, 0, 0, 0, 0],
        ];

        assert_eq!(
            all_square_values(&square).collect_vec(),
            vec![1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn test_vector_permutations() {
        let expected = vec![vec![1, 2, 3, 4, 5], vec![1, 2, 5, 4, 3]];
        let actual = vector_permutations(&[2, 4], &[1, 2, 3, 4, 5]).collect_vec();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_align_vector() {
        let aligned = align_vector(&[(1, 3), (4, 0)], &[1, 2, 3, 4, 5]);
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

    #[test]
    fn test_read_write_square() -> std::io::Result<()> {
        let square: Square = [
            [1, 22, 21, 18, 3],
            [20, 2, 15, 9, 19],
            [14, 16, 13, 10, 12],
            [7, 17, 11, 24, 6],
            [23, 8, 5, 4, 25],
        ];

        let mut buf = Vec::new();

        write_square(&square, &mut buf)?;

        let square2 = read_square(&mut buf.as_slice())?;

        assert_eq!(square, square2);

        Ok(())
    }
}
