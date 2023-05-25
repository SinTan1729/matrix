use num::{traits::Zero, Integer};
use std::{
    fmt::{self, Debug, Display, Formatter},
    ops::{Add, Mul, Sub},
    result::Result,
};

#[derive(PartialEq, Debug, Clone)]
pub struct Matrix<T: Mul + Add + Sub> {
    entries: Vec<Vec<T>>,
}

impl<T: Mul + Add + Sub + Zero> Matrix<T> {
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

    pub fn height(&self) -> usize {
        self.entries.len()
    }

    pub fn width(&self) -> usize {
        self.entries[0].len()
    }

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

    pub fn rows(&self) -> Vec<Vec<T>>
    where
        T: Copy,
    {
        self.entries.clone()
    }

    pub fn columns(&self) -> Vec<Vec<T>>
    where
        T: Copy,
    {
        self.transpose().entries
    }

    pub fn is_square(&self) -> bool {
        self.height() == self.width()
    }

    pub fn submatrix(&self, i: usize, j: usize) -> Self
    where
        T: Copy,
    {
        let mut out = Vec::new();
        for (m, row) in self.rows().iter().enumerate() {
            if m == i {
                continue;
            }
            let mut new_row = Vec::new();
            for (n, entry) in row.iter().enumerate() {
                if n != j {
                    new_row.push(*entry);
                }
            }
            out.push(new_row);
        }
        Matrix { entries: out }
    }

    pub fn det(&self) -> Result<T, &'static str>
    where
        T: Copy,
        T: Mul<Output = T>,
        T: Sub<Output = T>,
    {
        if self.is_square() {
            let out = if self.width() == 1 {
                self.entries[0][0]
            } else {
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
}

impl<T: Debug + Mul + Add + Sub> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{:?}", self.entries)
    }
}

impl<T: Mul<Output = T> + Add + Sub + Copy + Zero> Mul for Matrix<T> {
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
