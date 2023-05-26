//! This is a crate for very basic matrix operations
//! with any type that supports addition, substraction,
//! and multiplication. Additional properties might be
//! needed for certain operations.
//! I created it mostly to learn using generic types
//! and traits.
//!
//! Sayantan Santra (2023)

use num::{
    traits::{One, Zero},
    Integer,
};
use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::{Add, Div, Mul, Neg, Sub},
    result::Result,
};

mod tests;

/// A generic matrix struct (over any type with addition, substraction
/// and multiplication defined on it).
/// Look at [`from`](Self::from()) to see examples.
#[derive(PartialEq, Debug, Clone)]
pub struct Matrix<T: Mul + Add + Sub> {
    entries: Vec<Vec<T>>,
}

impl<T: Mul + Add + Sub> Matrix<T> {
    /// Creates a matrix from given 2D "array" in a `Vec<Vec<T>>` form.
    /// It'll throw an error if all the given rows aren't of the same size.
    /// # Example
    /// ```
    /// use matrix_basic::Matrix;
    /// let m = Matrix::from(vec![vec![1,2,3], vec![4,5,6]]);
    /// ```
    /// will create the following matrix:  
    /// ⌈1,2,3⌉  
    /// ⌊4,5,6⌋
    pub fn from(entries: Vec<Vec<T>>) -> Result<Matrix<T>, &'static str> {
        let mut equal_rows = true;
        let row_len = entries[0].len();
        for row in &entries {
            if row_len != row.len() {
                equal_rows = false;
                break;
            }
        }
        if equal_rows {
            Ok(Matrix { entries })
        } else {
            Err("Unequal rows.")
        }
    }

    /// Returns the height of a matrix.
    pub fn height(&self) -> usize {
        self.entries.len()
    }

    /// Returns the width of a matrix.
    pub fn width(&self) -> usize {
        self.entries[0].len()
    }

    /// Returns the transpose of a matrix.
    pub fn transpose(&self) -> Self
    where
        T: Copy,
    {
        let mut out = Vec::new();
        for i in 0..self.width() {
            let mut column = Vec::new();
            for row in &self.entries {
                column.push(row[i]);
            }
            out.push(column)
        }
        Matrix { entries: out }
    }

    /// Returns a reference to the rows of a matrix as `&Vec<Vec<T>>`.
    pub fn rows(&self) -> &Vec<Vec<T>> {
        &self.entries
    }

    /// Return the columns of a matrix as `Vec<Vec<T>>`.
    pub fn columns(&self) -> Vec<Vec<T>>
    where
        T: Copy,
    {
        self.transpose().entries
    }

    /// Return true if a matrix is square and false otherwise.
    pub fn is_square(&self) -> bool {
        self.height() == self.width()
    }

    /// Returns a matrix after removing the provided row and column from it.
    /// Note: Row and column numbers are 0-indexed.
    /// # Example
    /// ```
    /// use matrix_basic::Matrix;
    /// let m = Matrix::from(vec![vec![1,2,3],vec![4,5,6]]).unwrap();
    /// let n = Matrix::from(vec![vec![5,6]]).unwrap();
    /// assert_eq!(m.submatrix(0,0),n);
    /// ```
    pub fn submatrix(&self, row: usize, col: usize) -> Self
    where
        T: Copy,
    {
        let mut out = Vec::new();
        for (m, row_iter) in self.entries.iter().enumerate() {
            if m == row {
                continue;
            }
            let mut new_row = Vec::new();
            for (n, entry) in row_iter.iter().enumerate() {
                if n != col {
                    new_row.push(*entry);
                }
            }
            out.push(new_row);
        }
        Matrix { entries: out }
    }

    /// Returns the determinant of a square matrix. This method additionally requires [`Zero`],
    /// [`One`] and [`Copy`] traits. Also, we need that the [`Mul`] and [`Add`] operations
    /// return the same type `T`. This uses basic recursive algorithm using cofactor-minor.
    /// See [`det_in_field`](Self::det_in_field()) for faster determinant calculation in fields.
    /// It'll throw an error if the provided matrix isn't square.
    /// # Example
    /// ```
    /// use matrix_basic::Matrix;
    /// let m = Matrix::from(vec![vec![1,2],vec![3,4]]).unwrap();
    /// assert_eq!(m.det(),Ok(-2));
    /// ```
    pub fn det(&self) -> Result<T, &'static str>
    where
        T: Copy,
        T: Mul<Output = T>,
        T: Sub<Output = T>,
        T: Zero,
    {
        if self.is_square() {
            // It's a recursive algorithm using minors.
            // TODO: Implement a faster algorithm.
            let out = if self.width() == 1 {
                self.entries[0][0]
            } else {
                // Add the minors multiplied by cofactors.
                let n = 0..self.width();
                let mut out = T::zero();
                for i in n {
                    if i.is_even() {
                        out = out + (self.entries[0][i] * self.submatrix(0, i).det().unwrap());
                    } else {
                        out = out - (self.entries[0][i] * self.submatrix(0, i).det().unwrap());
                    }
                }
                out
            };
            Ok(out)
        } else {
            Err("Provided matrix isn't square.")
        }
    }

    /// Returns the determinant of a square matrix over a field i.e. needs [`One`] and [`Div`] traits.
    /// See [`det`](Self::det()) for determinants in rings.
    /// This method uses row reduction as is much faster.
    /// It'll throw an error if the provided matrix isn't square.
    /// # Example
    /// ```
    /// use matrix_basic::Matrix;
    /// let m = Matrix::from(vec![vec![1.0,2.0],vec![3.0,4.0]]).unwrap();
    /// assert_eq!(m.det(),Ok(-2.0));
    /// ```
    pub fn det_in_field(&self) -> Result<T, &'static str>
    where
        T: Copy,
        T: Mul<Output = T>,
        T: Sub<Output = T>,
        T: Zero,
        T: One,
        T: PartialEq,
        T: Div<Output = T>,
    {
        if self.is_square() {
            // Cloning is necessary as we'll be doing row operations on it.
            let mut rows = self.entries.clone();
            let mut multiplier = T::one();
            let h = self.height();
            let w = self.width();
            for i in 0..h {
                // First check if the row has diagonal element 0, if yes, then swap.
                if rows[i][i] == T::zero() {
                    let mut zero_column = true;
                    for j in (i + 1)..h {
                        if rows[j][i] != T::zero() {
                            rows.swap(i, j);
                            multiplier = T::zero() - multiplier;
                            zero_column = false;
                            break;
                        }
                    }
                    if zero_column {
                        return Ok(T::zero());
                    }
                }
                for j in (i + 1)..h {
                    let ratio = rows[j][i] / rows[i][i];
                    for k in i..w {
                        rows[j][k] = rows[j][k] - rows[i][k] * ratio;
                    }
                }
            }
            for (i, row) in rows.iter().enumerate() {
                multiplier = multiplier * row[i];
            }
            Ok(multiplier)
        } else {
            Err("Provided matrix isn't square.")
        }
    }

    /// Returns the row echelon form of a matrix over a field i.e. needs [`One`] and [`Div`] traits.
    /// # Example
    /// ```
    /// use matrix_basic::Matrix;
    /// let m = Matrix::from(vec![vec![1.0,2.0,3.0],vec![3.0,4.0,5.0]]).unwrap();
    /// let n = Matrix::from(vec![vec![1.0,2.0,3.0], vec![0.0,-2.0,-4.0]]).unwrap();
    /// assert_eq!(m.row_echelon(),n);
    /// ```
    pub fn row_echelon(&self) -> Self
    where
        T: Copy,
        T: Mul<Output = T>,
        T: Sub<Output = T>,
        T: Zero,
        T: One,
        T: PartialEq,
        T: Div<Output = T>,
    {
        // Cloning is necessary as we'll be doing row operations on it.
        let mut rows = self.entries.clone();
        let mut offset = 0;
        let h = self.height();
        let w = self.width();
        for i in 0..h {
            // Check if all the rows below are 0
            if i + offset >= self.width() {
                break;
            }
            // First check if the row has diagonal element 0, if yes, then swap.
            if rows[i][i + offset] == T::zero() {
                let mut zero_column = true;
                for j in (i + 1)..h {
                    if rows[j][i + offset] != T::zero() {
                        rows.swap(i, j);
                        zero_column = false;
                        break;
                    }
                }
                if zero_column {
                    offset += 1;
                }
            }
            for j in (i + 1)..h {
                let ratio = rows[j][i + offset] / rows[i][i + offset];
                for k in (i + offset)..w {
                    rows[j][k] = rows[j][k] - rows[i][k] * ratio;
                }
            }
        }
        Matrix { entries: rows }
    }

    /// Returns the column echelon form of a matrix over a field i.e. needs [`One`] and [`Div`] traits.
    /// It's just the transpose of the row echelon form of the transpose.
    /// See [`row_echelon`](Self::row_echelon()) and [`transpose`](Self::transpose()).
    pub fn column_echelon(&self) -> Self
    where
        T: Copy,
        T: Mul<Output = T>,
        T: Sub<Output = T>,
        T: Zero,
        T: One,
        T: PartialEq,
        T: Div<Output = T>,
    {
        self.transpose().row_echelon().transpose()
    }

    /// Returns the reduced row echelon form of a matrix over a field i.e. needs [`One`] and [`Div`] traits.
    /// # Example
    /// ```
    /// use matrix_basic::Matrix;
    /// let m = Matrix::from(vec![vec![1.0,2.0,3.0],vec![3.0,4.0,5.0]]).unwrap();
    /// let n = Matrix::from(vec![vec![1.0,2.0,3.0], vec![0.0,1.0,2.0]]).unwrap();
    /// assert_eq!(m.reduced_row_echelon(),n);
    /// ```
    pub fn reduced_row_echelon(&self) -> Self
    where
        T: Copy,
        T: Mul<Output = T>,
        T: Sub<Output = T>,
        T: Zero,
        T: One,
        T: PartialEq,
        T: Div<Output = T>,
    {
        let mut echelon = self.row_echelon();
        let mut offset = 0;
        for row in &mut echelon.entries {
            while row[offset] == T::zero() {
                offset += 1;
            }
            let divisor = row[offset];
            for entry in row.iter_mut().skip(offset) {
                *entry = *entry / divisor;
            }
            offset += 1;
        }
        echelon
    }

    /// Creates a zero matrix of a given size.
    pub fn zero(height: usize, width: usize) -> Self
    where
        T: Zero,
    {
        let mut out = Vec::new();
        for _ in 0..height {
            let mut new_row = Vec::new();
            for _ in 0..width {
                new_row.push(T::zero());
            }
            out.push(new_row);
        }
        Matrix { entries: out }
    }

    /// Creates an identity matrix of a given size.
    pub fn identity(size: usize) -> Self
    where
        T: Zero,
        T: One,
    {
        let mut out = Vec::new();
        for i in 0..size {
            let mut new_row = Vec::new();
            for j in 0..size {
                if i == j {
                    new_row.push(T::one());
                } else {
                    new_row.push(T::zero());
                }
            }
            out.push(new_row);
        }
        Matrix { entries: out }
    }

    // TODO: Canonical forms, eigenvalues, eigenvectors etc.
}

impl<T: Debug + Mul + Add + Sub> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}", self.entries)
    }
}

impl<T: Mul<Output = T> + Add + Sub + Copy + Zero> Mul for Matrix<T> {
    // TODO: Implement a faster algorithm.
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        let width = self.width();
        if width != other.height() {
            panic!("Row length of first matrix must be same as column length of second matrix.");
        } else {
            let mut out = Vec::new();
            for row in self.rows() {
                let mut new_row = Vec::new();
                for col in other.columns() {
                    let mut prod = row[0] * col[0];
                    for i in 1..width {
                        prod = prod + (row[i] * col[i]);
                    }
                    new_row.push(prod)
                }
                out.push(new_row);
            }
            Matrix { entries: out }
        }
    }
}

impl<T: Add<Output = T> + Sub + Mul + Copy + Zero> Add for Matrix<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        if self.height() == other.height() && self.width() == other.width() {
            let mut out = self.entries.clone();
            for (i, row) in self.rows().iter().enumerate() {
                for (j, entry) in other.rows()[i].iter().enumerate() {
                    out[i][j] = row[j] + *entry;
                }
            }
            Matrix { entries: out }
        } else {
            panic!("Both matrices must be of same dimensions.");
        }
    }
}

impl<T: Add + Sub<Output = T> + Mul + Copy + Neg<Output = T>> Neg for Matrix<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut out = self;
        for row in &mut out.entries {
            for entry in row {
                *entry = -*entry;
            }
        }
        out
    }
}

impl<T: Add + Sub<Output = T> + Mul + Copy + Zero + Neg<Output = T>> Sub for Matrix<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        if self.height() == other.height() && self.width() == other.width() {
            self + -other
        } else {
            panic!("Both matrices must be of same dimensions.");
        }
    }
}
