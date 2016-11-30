# Compressed Column Storage

Many sparse matrix methods use **Compressed Column Storage** (CCS) format for
representing matrices.

\begin{equation}
  \begin{bmatrix}
    1 & 0 & 4 & 0 \\
    0 & 5 & 0 & 2 \\
    0 & 0 & 0 & 1 \\
    6 & 8 & 0 & 0
  \end{bmatrix}
\end{equation}

    val      =  1   6   5   8   4   2   1
    row_ind  =  0   3   1   3   0   1   2
    col_ptr  =  0       2       4   5       7

- `val`: Each nonzero entry in the array is placed in the `val` vector, entries
  are ordered by column first, then by row within the column.
- `col_ptr`: Has one entry for each column, plus a last entry which is the
  number of total nonzeros (nnz). The `col_ptr` entry for a column points to the
  start of the column in `val` (e.g. 5 is the 3rd value in `val` denoting the
  start of the 2nd column).
- `row_ind`: Row index of each entry within a column.

CCS format is storage efficient, but is difficult to create incrementally,
since each new nonzero addition to a column causes a shift in all subsequent
entries, making it inefficient to create dynamically.
