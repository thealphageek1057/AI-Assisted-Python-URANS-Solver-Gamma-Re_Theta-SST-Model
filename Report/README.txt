====================================================================
 README: Transitional Airfoil CFD Solver using γ–Re_θ–SST (Python)
====================================================================

Project Title:
--------------
AI-Assisted Modular CFD Solver for 2D Transitional Flow over NACA Airfoils

Author:
-------
Vigneshwara Koka Balaji, Matriculation No. 71624
CMS@TUBAF, TU Bergakadamie Freiberg

Description:
------------
This project implements a fully implicit, modular Python solver for simulating 
incompressible transitional flows over NACA airfoils using the URANS framework 
with the γ–Re_θ transition model and SST k–ω turbulence closure.

The solver uses:
- Crank–Nicolson θ-scheme (second-order implicit time integration)
- Structured FVM on multizone mesh
- Sparse matrix solvers (BiCGSTAB)
- Automatic boundary condition tagging
- Live aerodynamic coefficient (CL, CD) computation
- Wake animation and visualization of transition dynamics

Main Code:
----------
- `urans_solver_re_theta_SST.ipynb`  → Full solver and postprocessing pipeline

How to Run:
-----------
1. Open `urans_solver_re_theta_SST.ipynb` in Jupyter.
2. Execute the cells sequentially.
3. Input the Reynolds number and angle of attack when prompted.
4. Simulation snapshots, plots, and diagnostics will be generated automatically.

Output:
-------
Upon execution, the following are created:

(1) `snapshots/` folder — contains:
    - Velocity magnitude plots (Vmag_step_XXXX.png)
    - Pressure coefficient plots (CP_step_XXXX.png)
    - Streamline and transition animations

(2) Diagnostic plots in root:
    - `cl_vs_time_plot.png`       → Lift coefficient over time
    - `cd_vs_time_plot.png`       → Drag coefficient over time
    - `clcd_vs_time_plot.png`     → CL/CD ratio over time
    - `wake_with_separation.gif`  → Animated streamline and wake field

(3) All outputs are used in the final report for validation and discussion.

Known Limitations:
------------------
- CL and CD values may be physically inconsistent at early timesteps due to 
  numerical oscillations or insufficient outlet resolution.
- The solver currently supports structured mesh only.
- The transition model is highly sensitive to boundary conditions and timestep size.

Acknowledgments:
----------------
This project was developed under the AI-Assisted Programming 2025 framework,
with guidance from Prof. B. Eidel and assistance from OpenAI ChatGPT-4-0-mini-high.

Contact:
--------
For questions, contact:
Vigneshwara.Koka-Balaji@student.tu-freiberg.de
