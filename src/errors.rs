use std::{
    error::Error,
    fmt::{self, Display, Formatter},
};

/// Error type for using in this crate. Mostly to reduce writing
/// error description every time.
#[derive(Debug, PartialEq)]
pub enum MatrixError {
    /// Provided matrix isn't square.
    NotSquare,
    /// provided matrix is singular.
    Singular,
    /// Provided array has unequal rows.
    UnequalRows,
}

impl Display for MatrixError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let out = match *self {
            Self::NotSquare => "provided matrix isn't square",
            Self::Singular => "provided matrix is singular",
            Self::UnequalRows => "provided array has unequal rows",
        };
        write!(f, "{out}")
    }
}

impl Error for MatrixError {}
