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
    ops::{Add, Mul, Sub},
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

    /// Return the height of a matrix.
    pub fn height(&self) -> usize {
        self.entries.len()
    }

    /// Return the width of a matrix.
    pub fn width(&self) -> usize {
        self.entries[0].len()
    }

    /// Return the transpose of a matrix.
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

    /// Return a reference to the rows of a matrix as `&Vec<Vec<T>>`.
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

    /// Return a matrix after removing the provided row and column from it.
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

    /// Return the determinant of a square matrix. This method additionally requires [`Zero`],
    /// [`One`] and [`Copy`] traits. Also, we need that the [`Mul`] and [`Add`] operations
    /// return the same type `T`.
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
            // TODO: Implement a faster algorithm. Maybe use row reduction for fields.
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
}

impl<T: Debug + Mul + Add + Sub> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}", self.entries)
    }
}

impl<T: Mul<Output = T> + Add + Sub + Copy + Zero> Mul for Matrix<T> {
    // TODO: Implement a faster algorithm. Maybe use row reduction for fields.
    type Output = Self;
    fn mul(self, other: Self) -> Self {
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
    fn add(self, other: Self) -> Self {
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

impl<T: Add + Sub<Output = T> + Mul + Copy + Zero> Sub for Matrix<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        if self.height() == other.height() && self.width() == other.width() {
            let mut out = self.entries.clone();
            for (i, row) in self.rows().iter().enumerate() {
                for (j, entry) in other.rows()[i].iter().enumerate() {
                    out[i][j] = row[j] - *entry;
                }
            }
            Matrix { entries: out }
        } else {
            panic!("Both matrices must be of same dimensions.");
        }
    }
}
