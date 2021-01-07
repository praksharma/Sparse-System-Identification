# System-Identification
1. The "sparse dynamics" directory is the original code, also available at https://www.pnas.org/content/113/15/3932
Sparse Identification of Nonlinear Dynamics (SINDY)
Copyright 2015, All Rights Reserved
Code by Steven L. Brunton (sbrunton@uw.edu)
For Paper, "Discovering Governing Equations from Data by 
        Sparse Identification of Nonlinear Dynamical Systems" 
Proceedings of the National Academy of Sciences
Vol. 113, No. 15, pp. 3932—3937, 2016.
by S. L. Brunton, J. L. Proctor, and J. N. Kutz


2. "Python code" directory contains the translated code from MATLAB. 
These codes worked well except for the adjustments that were made to entertain numpy 1D-array (for 1D problem) in a function that accepts a numpy matrix/ 2D array (for 2D or 3D problem)

3. "Improved" directory contains an improved version of the Python code
The common function to entertain both 1D and 2D numpy array were seperated to make the code more stable and reliable.
